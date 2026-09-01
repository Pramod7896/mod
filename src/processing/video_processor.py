"""End-to-end video detection, ByteTrack tracking, counting, and annotation."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.analytics.metrics import (
    SummaryMetrics,
    build_timeline,
    calculate_summary,
    events_to_dataframe,
)
from src.config import ProcessingConfig
from src.counting.line_counter import LineCounter
from src.detection.detector import TrackedDetection, YOLODetector
from src.tracking.tracker import TrackHistory
from src.utils.file_utils import output_video_path
from src.utils.video_utils import VideoInfo, convert_to_h264, format_timestamp, get_video_info

logger = logging.getLogger(__name__)


@dataclass
class LiveStats:
    """Live processing stats emitted to the Streamlit UI."""

    frame_number: int
    total_frames: int
    total_count: int
    forward_count: int
    reverse_count: int
    objects_per_minute: float
    average_confidence: float
    processing_fps: float
    annotated_frame: np.ndarray
    events: list[dict]


@dataclass
class ProcessingResult:
    """Final result returned after analysis."""

    video_info: VideoInfo
    output_video: Path
    events_df: pd.DataFrame
    timeline_df: pd.DataFrame
    summary: SummaryMetrics
    class_distribution: pd.DataFrame
    confidence_values: list[float]
    warning: str | None = None


class VideoProcessor:
    """Processes manufacturing videos using YOLO + ByteTrack + line counting."""

    def __init__(self, detector: YOLODetector, config: ProcessingConfig) -> None:
        self.detector = detector
        self.config = config
        self.events: list[dict] = []
        self.confidences: list[float] = []
        self.crossing_highlights: dict[int, int] = {}

    def process(
        self,
        input_path: Path,
        progress_callback: Callable[[LiveStats], None] | None = None,
    ) -> ProcessingResult:
        """Process a video, write annotated output, and return analytics."""

        info = get_video_info(input_path)
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise ValueError("The uploaded video could not be opened.")

        raw_output = output_video_path()
        writer = cv2.VideoWriter(
            str(raw_output),
            cv2.VideoWriter_fourcc(*"mp4v"),
            info.fps,
            (info.width, info.height),
        )
        if not writer.isOpened():
            cap.release()
            raise RuntimeError("Could not initialize the MP4 video writer.")

        counter = LineCounter(
            mode=self.config.counting_mode,
            line_position=self.config.line_position,
            direction_filter=self.config.count_direction,
            frame_width=info.width,
            frame_height=info.height,
        )
        history = TrackHistory()
        start_time = time.perf_counter()
        frame_number = 0
        last_detections: list[TrackedDetection] = []

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_number += 1

                run_detection = self.config.frame_skip == 0 or (frame_number - 1) % (self.config.frame_skip + 1) == 0
                detections = (
                    self.detector.track_frame(
                        frame=frame,
                        confidence=self.config.confidence_threshold,
                        iou=self.config.iou_threshold,
                        image_size=self.config.input_resolution,
                        selected_classes=self.config.selected_classes,
                    )
                    if run_detection
                    else last_detections
                )
                if run_detection:
                    last_detections = detections

                self._update_counts(detections, history, counter, frame_number, info.fps)
                annotated = self._annotate_frame(frame, detections, history, counter, frame_number, start_time)
                writer.write(annotated)

                if progress_callback and (frame_number == 1 or frame_number % max(1, int(info.fps // 2)) == 0):
                    elapsed_video = frame_number / info.fps
                    elapsed_processing = max(time.perf_counter() - start_time, 1e-6)
                    recent_events = [event for event in self.events if elapsed_video - event["seconds"] <= 60]
                    progress_callback(
                        LiveStats(
                            frame_number=frame_number,
                            total_frames=info.frame_count,
                            total_count=counter.total_count,
                            forward_count=counter.forward_count,
                            reverse_count=counter.reverse_count,
                            objects_per_minute=float(len(recent_events)),
                            average_confidence=float(np.mean(self.confidences)) if self.confidences else 0.0,
                            processing_fps=frame_number / elapsed_processing,
                            annotated_frame=cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                            events=list(self.events),
                        )
                    )
        finally:
            cap.release()
            writer.release()

        processing_time = time.perf_counter() - start_time
        final_output = raw_output.with_name(raw_output.stem + "_h264.mp4")
        output_path, warning = convert_to_h264(raw_output, final_output)
        events_df = events_to_dataframe(self.events)
        timeline = build_timeline(events_df, info.duration_seconds)
        summary = calculate_summary(
            events_df=events_df,
            timeline_df=timeline,
            forward_count=counter.forward_count,
            reverse_count=counter.reverse_count,
            processing_time_seconds=processing_time,
            elapsed_video_seconds=info.duration_seconds,
        )
        class_distribution = (
            events_df.groupby("class").size().reset_index(name="count").sort_values("count", ascending=False)
            if not events_df.empty
            else pd.DataFrame(columns=["class", "count"])
        )
        return ProcessingResult(
            video_info=info,
            output_video=output_path,
            events_df=events_df,
            timeline_df=timeline,
            summary=summary,
            class_distribution=class_distribution,
            confidence_values=list(self.confidences),
            warning=warning,
        )

    def _update_counts(
        self,
        detections: list[TrackedDetection],
        history: TrackHistory,
        counter: LineCounter,
        frame_number: int,
        fps: float,
    ) -> None:
        for detection in detections:
            previous_center = history.update(detection.track_id, detection.center)
            self.confidences.append(detection.confidence)
            direction = counter.crossing_direction(previous_center, detection.center)
            if counter.should_count(detection.track_id, direction):
                assert direction is not None
                counter.register(detection.track_id, direction)
                self.crossing_highlights[detection.track_id] = 15
                seconds = frame_number / fps
                self.events.append(
                    {
                        "timestamp": format_timestamp(seconds),
                        "seconds": seconds,
                        "frame_number": frame_number,
                        "track_id": detection.track_id,
                        "class": detection.class_name,
                        "confidence": detection.confidence,
                        "direction": direction,
                    }
                )

    def _annotate_frame(
        self,
        frame: np.ndarray,
        detections: list[TrackedDetection],
        history: TrackHistory,
        counter: LineCounter,
        frame_number: int,
        start_time: float,
    ) -> np.ndarray:
        annotated = frame.copy()
        if self.config.show_counting_line:
            self._draw_counting_line(annotated, counter)
        for detection in detections:
            self._draw_detection(annotated, detection, history)
        self._draw_overlay(annotated, counter, frame_number, start_time)
        self.crossing_highlights = {
            track_id: ttl - 1 for track_id, ttl in self.crossing_highlights.items() if ttl > 1
        }
        return annotated

    def _draw_counting_line(self, frame: np.ndarray, counter: LineCounter) -> None:
        line = counter.line_coordinate()
        h, w = frame.shape[:2]
        color = (255, 210, 40)
        if counter.mode == "ROI Zone":
            x1, y1, x2, y2 = counter.roi_bounds()
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 210, 40), -1)
            cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            cv2.putText(frame, "COUNTING ZONE", (18, max(34, y1 - 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            return
        if counter.mode == "Vertical Line":
            cv2.line(frame, (line, 0), (line, h), color, 3)
            cv2.putText(frame, "COUNTING ZONE", (line + 12, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        else:
            cv2.line(frame, (0, line), (w, line), color, 3)
            cv2.putText(frame, "COUNTING ZONE", (18, max(34, line - 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    def _draw_detection(self, frame: np.ndarray, detection: TrackedDetection, history: TrackHistory) -> None:
        x1, y1, x2, y2 = detection.bbox
        color = self._stable_color(detection.class_name)
        if detection.track_id in self.crossing_highlights:
            color = (45, 255, 120)
            thickness = 4
        else:
            thickness = 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        label_parts = [detection.class_name.title()]
        if self.config.show_tracking_ids:
            label_parts.append(f"ID #{detection.track_id}")
        if self.config.show_confidence:
            label_parts.append(f"{detection.confidence * 100:.1f}%")
        label = " | ".join(label_parts)
        self._draw_label(frame, label, x1, y1, color)

        if self.config.show_trail:
            points = history.get(detection.track_id)
            for p1, p2 in zip(points, points[1:], strict=False):
                cv2.line(frame, p1, p2, color, 2)
            cv2.circle(frame, detection.center, 4, color, -1)

    def _draw_overlay(self, frame: np.ndarray, counter: LineCounter, frame_number: int, start_time: float) -> None:
        elapsed = max(time.perf_counter() - start_time, 1e-6)
        fps = frame_number / elapsed
        recent_events = [event for event in self.events if frame_number - event["frame_number"] <= 150]
        avg_conf = float(np.mean(self.confidences)) * 100 if self.confidences else 0.0
        lines = [
            "LIVE PRODUCTION",
            f"Count: {counter.total_count}",
            f"Forward: {counter.forward_count}  Reverse: {counter.reverse_count}",
            f"Rate: {len(recent_events)} / min",
            f"Avg Conf: {avg_conf:.1f}%",
        ]
        if self.config.show_fps:
            lines.append(f"Processing FPS: {fps:.1f}")

        x, y, width, height = 18, 18, 330, 30 + len(lines) * 26
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + width, y + height), (10, 22, 34), -1)
        cv2.addWeighted(overlay, 0.68, frame, 0.32, 0, frame)
        cv2.rectangle(frame, (x, y), (x + width, y + height), (72, 210, 255), 1)
        for idx, text in enumerate(lines):
            color = (102, 232, 255) if idx == 0 else (235, 245, 255)
            cv2.putText(frame, text, (x + 14, y + 28 + idx * 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2)

    @staticmethod
    def _stable_color(name: str) -> tuple[int, int, int]:
        palette = [(74, 222, 128), (56, 189, 248), (250, 204, 21), (248, 113, 113), (167, 139, 250)]
        return palette[sum(ord(char) for char in name) % len(palette)]

    @staticmethod
    def _draw_label(frame: np.ndarray, label: str, x: int, y: int, color: tuple[int, int, int]) -> None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.58
        thickness = 2
        (text_width, text_height), _ = cv2.getTextSize(label, font, scale, thickness)
        top = max(0, y - text_height - 12)
        cv2.rectangle(frame, (x, top), (x + text_width + 12, top + text_height + 10), color, -1)
        cv2.putText(frame, label, (x + 6, top + text_height + 4), font, scale, (5, 16, 24), thickness)

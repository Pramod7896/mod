"""Detection and ByteTrack result normalization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TrackedDetection:
    """A single tracked object observation."""

    track_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]
    center: tuple[int, int]


class YOLODetector:
    """Thin adapter around Ultralytics YOLO tracking output."""

    def __init__(self, model: Any, device: str) -> None:
        self.model = model
        self.device = device

    def track_frame(
        self,
        frame: np.ndarray,
        confidence: float,
        iou: float,
        image_size: int,
        selected_classes: list[str] | None = None,
    ) -> list[TrackedDetection]:
        """Run YOLO + ByteTrack on one frame and return normalized detections."""

        results = self.model.track(
            source=frame,
            conf=confidence,
            iou=iou,
            imgsz=image_size,
            persist=True,
            tracker="bytetrack.yaml",
            device=self.device,
            verbose=False,
        )
        if not results:
            return []

        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or boxes.id is None:
            return []

        names = getattr(result, "names", {}) or getattr(self.model, "names", {})
        xyxy = boxes.xyxy.cpu().numpy()
        track_ids = boxes.id.int().cpu().tolist()
        class_ids = boxes.cls.int().cpu().tolist()
        confidences = boxes.conf.cpu().tolist()
        allowed = {name.lower() for name in selected_classes or [] if name}

        detections: list[TrackedDetection] = []
        for box, track_id, class_id, conf in zip(xyxy, track_ids, class_ids, confidences, strict=False):
            class_name = str(names.get(class_id, class_id) if isinstance(names, dict) else names[class_id])
            if allowed and class_name.lower() not in allowed:
                continue
            x1, y1, x2, y2 = [int(round(v)) for v in box]
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            detections.append(
                TrackedDetection(
                    track_id=int(track_id),
                    class_id=int(class_id),
                    class_name=class_name,
                    confidence=float(conf),
                    bbox=(x1, y1, x2, y2),
                    center=center,
                )
            )
        return detections

"""Video probing, codec handling, and time formatting helpers."""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2

from src.config import SAMPLES_DIR

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VideoInfo:
    """Basic metadata extracted from a video file."""

    path: Path
    width: int
    height: int
    fps: float
    frame_count: int
    duration_seconds: float
    file_size_mb: float


def get_video_info(path: Path) -> VideoInfo:
    """Open a video and return metadata, raising ValueError for invalid input."""

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError("The uploaded video could not be opened. Please try another file.")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if width <= 0 or height <= 0 or frame_count <= 0:
        raise ValueError("The uploaded video appears to have no readable frames.")
    safe_fps = fps if fps > 0 else 25.0
    return VideoInfo(
        path=path,
        width=width,
        height=height,
        fps=safe_fps,
        frame_count=frame_count,
        duration_seconds=frame_count / safe_fps,
        file_size_mb=path.stat().st_size / (1024 * 1024),
    )


def find_latest_downloads_video() -> Path | None:
    """Return the newest supported video file from the user's Downloads folder."""

    downloads = Path.home() / "Downloads"
    if not downloads.exists():
        return None
    patterns = ("*.mp4", "*.avi", "*.mov", "*.mkv")
    videos: list[Path] = []
    for pattern in patterns:
        videos.extend(downloads.glob(pattern))
    videos = [path for path in videos if path.is_file()]
    if not videos:
        return None
    return max(videos, key=lambda path: path.stat().st_mtime)


def find_sample_video() -> Path | None:
    """Return the bundled sample video first, then fall back to Downloads."""

    bundled = SAMPLES_DIR / "sample_conveyor.mp4"
    if bundled.exists():
        return bundled
    return find_latest_downloads_video()


def format_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""

    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def ffmpeg_available() -> bool:
    """Return True when ffmpeg is on PATH."""

    return shutil.which("ffmpeg") is not None


def convert_to_h264(input_path: Path, output_path: Path) -> tuple[Path, str | None]:
    """Convert an MP4 to browser-friendly H.264 when ffmpeg is available."""

    if not ffmpeg_available():
        return input_path, "ffmpeg was not found, so the OpenCV MP4 output was used directly."
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        input_path.unlink(missing_ok=True)
        return output_path, None
    except (subprocess.CalledProcessError, OSError) as exc:
        logger.warning("ffmpeg conversion failed: %s", exc)
        return input_path, "ffmpeg conversion failed, so the OpenCV MP4 output was used directly."

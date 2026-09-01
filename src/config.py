"""Application configuration for IndustrialEye AI."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"
TEMP_DIR = ROOT_DIR / "temp"
OUTPUTS_DIR = ROOT_DIR / "outputs"
ASSETS_DIR = ROOT_DIR / "assets"

DEFAULT_MODEL_NAME = "yolo11n.pt"
OPEN_VOCAB_MODEL_NAME = "yolov8s-worldv2.pt"
CUSTOM_MODEL_NAME = "manufacturing_best.pt"

VIDEO_EXTENSIONS = ("mp4", "avi", "mov", "mkv")


@dataclass(frozen=True)
class ProcessingConfig:
    """Runtime options selected from the Streamlit sidebar."""

    model_mode: str = "Standard YOLO"
    model_path: str = DEFAULT_MODEL_NAME
    target_labels: list[str] = field(default_factory=list)
    selected_classes: list[str] = field(default_factory=list)
    confidence_threshold: float = 0.35
    iou_threshold: float = 0.5
    counting_mode: str = "Horizontal Line"
    line_position: float = 0.5
    count_direction: str = "Both"
    frame_skip: int = 0
    input_resolution: int = 640
    show_tracking_ids: bool = True
    show_confidence: bool = True
    show_trail: bool = True
    show_counting_line: bool = True
    show_fps: bool = True
    debug: bool = False


def ensure_directories() -> None:
    """Create all runtime directories if missing."""

    for path in (MODELS_DIR, TEMP_DIR, OUTPUTS_DIR, ASSETS_DIR):
        path.mkdir(parents=True, exist_ok=True)

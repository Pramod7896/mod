"""Ultralytics model loading with Streamlit cache support."""

from __future__ import annotations

import logging
import os
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from src.config import DEFAULT_MODEL_NAME, MODELS_DIR, OPEN_VOCAB_MODEL_NAME

logger = logging.getLogger(__name__)

_LOCAL_YOLO_CONFIG = Path(__file__).resolve().parents[2] / ".ultralytics"
_LOCAL_YOLO_CONFIG.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(_LOCAL_YOLO_CONFIG))
warnings.filterwarnings("ignore", message="urllib3 .*doesn't match a supported version.*")
try:
    from requests.exceptions import RequestsDependencyWarning

    warnings.filterwarnings("ignore", category=RequestsDependencyWarning)
except Exception:
    pass

from ultralytics import YOLO  # noqa: E402 - config directory must be set first.


@contextmanager
def _temporary_cwd(path: Path):
    old_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_cwd)


@dataclass(frozen=True)
class DeviceInfo:
    """Resolved inference device details."""

    device: str
    label: str


def get_device_info() -> DeviceInfo:
    """Use CUDA when available, otherwise CPU."""

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        return DeviceInfo(device="cuda:0", label=f"NVIDIA GPU: {name}")
    return DeviceInfo(device="cpu", label="CPU")


def model_path_for_mode(model_mode: str, custom_path: Path | None = None) -> Path | str:
    """Resolve a model identifier/path for the selected mode."""

    if model_mode == "Custom Model":
        return custom_path if custom_path else MODELS_DIR / "manufacturing_best.pt"
    if model_mode == "Open Vocabulary":
        return MODELS_DIR / OPEN_VOCAB_MODEL_NAME
    return MODELS_DIR / DEFAULT_MODEL_NAME


def load_yolo_model(model_source: Path | str) -> Any:
    """Load a YOLO model. Ultralytics downloads official weights on first use."""

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if isinstance(model_source, Path) and model_source.parent == MODELS_DIR and not model_source.exists():
        if model_source.name not in {DEFAULT_MODEL_NAME, OPEN_VOCAB_MODEL_NAME}:
            raise FileNotFoundError(f"Custom model not found: {model_source}")
        with _temporary_cwd(MODELS_DIR):
            logger.info("Downloading/loading official YOLO weights %s under models/", model_source.name)
            return YOLO(model_source.name)
    source = str(model_source)
    logger.info("Loading YOLO model from %s", source)
    return YOLO(source)


def get_model_classes(model: Any) -> list[str]:
    """Return model class names as a sorted list."""

    names = getattr(model, "names", {}) or {}
    if isinstance(names, dict):
        return [str(names[key]) for key in sorted(names)]
    return [str(name) for name in names]


def try_set_open_vocab_classes(model: Any, labels: list[str]) -> tuple[bool, str | None]:
    """Set YOLO-World prompt classes if supported by the installed package."""

    clean_labels = [label.strip() for label in labels if label.strip()]
    if not clean_labels:
        return False, "Enter at least one target object label for open-vocabulary detection."
    if not hasattr(model, "set_classes"):
        return False, "This Ultralytics version/model does not expose set_classes()."
    try:
        model.set_classes(clean_labels)
        return True, None
    except Exception as exc:  # noqa: BLE001 - converted into a user-facing warning.
        logger.warning("Unable to set YOLO-World classes: %s", exc)
        return False, f"Open-vocabulary prompts could not be applied: {exc}"

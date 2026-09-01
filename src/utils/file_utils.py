"""Safe file helpers for uploads, outputs, and downloads."""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import BinaryIO

from src.config import OUTPUTS_DIR, TEMP_DIR


def unique_path(directory: Path, suffix: str, prefix: str) -> Path:
    """Return a non-existing path with a UUID file name."""

    directory.mkdir(parents=True, exist_ok=True)
    clean_suffix = suffix if suffix.startswith(".") else f".{suffix}"
    return directory / f"{prefix}_{uuid.uuid4().hex}{clean_suffix}"


def save_uploaded_file(uploaded_file: BinaryIO, suffix: str) -> Path:
    """Persist a Streamlit uploaded file into temp/ safely."""

    output_path = unique_path(TEMP_DIR, suffix=suffix, prefix="upload")
    with output_path.open("wb") as file:
        shutil.copyfileobj(uploaded_file, file)
    return output_path


def output_video_path() -> Path:
    """Create a unique processed-video path."""

    return unique_path(OUTPUTS_DIR, suffix=".mp4", prefix="processed")


def output_csv_path(name: str) -> Path:
    """Create a unique CSV path."""

    return unique_path(OUTPUTS_DIR, suffix=".csv", prefix=name)


def file_size_mb(path: Path) -> float:
    """Return file size in MB."""

    return path.stat().st_size / (1024 * 1024)


def cleanup_old_files(directory: Path, keep_last: int = 20) -> None:
    """Keep the newest runtime files and remove older leftovers."""

    files = sorted(
        [p for p in directory.glob("*") if p.is_file() and p.name != ".gitkeep"],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for path in files[keep_last:]:
        path.unlink(missing_ok=True)

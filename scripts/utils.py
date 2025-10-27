"""Utility helpers shared across scripts."""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable


def slugify(text: str, allow_slash: bool = False) -> str:
    """Convert text into a filesystem-friendly slug."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\-/]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    if not allow_slash:
        text = text.replace("/", "-")
    return text or "untitled"


def ensure_directory(path: Path | str) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def iter_chunks(iterable: Iterable, size: int):
    chunk = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def is_ci() -> bool:
    return os.environ.get("CI", "").lower() == "true"

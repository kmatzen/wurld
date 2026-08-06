"""wurld — posed sensor video in one playable WebM.

RGB video + per-frame camera pose + intrinsics + timestamps + bit-exact
uint16 signals (metric depth, confidence, IDs), carried by chromapakz tracks.
"""

from .container import (
    CAMERA_MODELS,
    Camera,
    Frame,
    Sequence,
    SignalMeta,
    info,
    read,
    write,
)
from . import conventions

__version__ = "0.1.0"

__all__ = [
    "Camera",
    "Frame",
    "Sequence",
    "SignalMeta",
    "CAMERA_MODELS",
    "read",
    "write",
    "info",
    "conventions",
]

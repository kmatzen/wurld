"""wurld — posed sensor video in one playable WebM.

RGB video + per-frame camera pose + intrinsics + timestamps + bit-exact
uint16 signals (metric depth, confidence, IDs), carried by chromapakz tracks.
"""

from .container import (
    CAMERA_MODELS,
    Camera,
    Frame,
    ImuStream,
    Sequence,
    SignalMeta,
    info,
    read,
    write,
)
from .stream import StreamReader, StreamWriter
from . import conventions

__version__ = "1.0.0"

__all__ = [
    "Camera",
    "Frame",
    "ImuStream",
    "Sequence",
    "SignalMeta",
    "CAMERA_MODELS",
    "read",
    "write",
    "info",
    "conventions",
    "StreamReader",
    "StreamWriter",
]

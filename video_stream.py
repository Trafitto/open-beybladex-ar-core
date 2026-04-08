"""
Non-blocking video capture using a background thread.

The reader thread continuously grabs frames from cv2.VideoCapture,
keeping only the latest one. The main loop calls read() to get the
most recent frame instantly, eliminating I/O wait from the webcam.

For video files, use cv2.VideoCapture directly (frame timing matters).
"""
import shutil
import subprocess
import threading
from typing import Optional, Tuple, Union

import cv2
import numpy as np


def is_live_source(src: Union[int, str]) -> bool:
    """True when *src* is a live/stream source (device index or network URL).

    Returns False for local file paths (.mp4, .avi, ...) which need
    frame-paced playback instead of non-blocking capture.
    """
    if isinstance(src, int):
        return True
    if isinstance(src, str):
        return src.startswith(("rtsp://", "rtmp://", "http://", "https://"))
    return False


# ---- PS3 Eye hardware configuration via v4l2-ctl -------------------------

_V4L2_CTL = shutil.which("v4l2-ctl")

# Primary name -> alternate names used by older kernel builds of gspca_ov534
_ALT_NAMES: dict[str, tuple[str, ...]] = {
    "gain_automatic": ("autogain",),
    "white_balance_automatic": ("auto_white_balance", "awb"),
    "red_balance": ("redblc",),
    "blue_balance": ("blueblc",),
    "horizontal_flip": ("hflip",),
    "vertical_flip": ("vflip",),
}

# Auto controls MUST come before their manual counterparts:
# gain_automatic must be set to 0 before gain becomes writable,
# auto_exposure must be set to 1 (Manual) before exposure becomes writable.
_PS3EYE_CTRL_MAP: list[tuple[str, str]] = [
    ("gain_automatic",            "PS3EYE_AUTOGAIN"),
    ("auto_exposure",             "PS3EYE_AUTO_EXPOSURE"),
    ("white_balance_automatic",   "PS3EYE_AUTO_WHITE_BALANCE"),
    ("exposure",                  "PS3EYE_EXPOSURE"),
    ("gain",                      "PS3EYE_GAIN"),
    ("brightness",                "PS3EYE_BRIGHTNESS"),
    ("contrast",                  "PS3EYE_CONTRAST"),
    ("saturation",                "PS3EYE_SATURATION_HW"),
    ("sharpness",                 "PS3EYE_SHARPNESS"),
    ("red_balance",               "PS3EYE_RED_BALANCE"),
    ("blue_balance",              "PS3EYE_BLUE_BALANCE"),
    ("horizontal_flip",           "PS3EYE_HFLIP"),
    ("vertical_flip",             "PS3EYE_VFLIP"),
    ("power_line_frequency",      "PS3EYE_POWER_LINE_FREQ"),
]


def _v4l2_set(device: str, ctrl: str, value: int) -> bool:
    if _V4L2_CTL is None:
        return False
    try:
        subprocess.run(
            [_V4L2_CTL, "-d", device, "--set-ctrl", f"{ctrl}={value}"],
            capture_output=True, timeout=2, check=True,
        )
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return False


def apply_ps3eye_settings(device_index: int) -> None:
    """Apply PS3 Eye hardware settings from config via v4l2-ctl."""
    import config

    if not getattr(config, "PS3EYE_ENABLED", False):
        return
    if _V4L2_CTL is None:
        print("PS3 Eye: v4l2-ctl not found -- install v4l2-utils")
        return

    dev = f"/dev/video{device_index}"
    applied, failed = [], []

    for v4l2_name, cfg_attr in _PS3EYE_CTRL_MAP:
        value = getattr(config, cfg_attr, None)
        if value is None:
            continue
        value = int(value)
        if _v4l2_set(dev, v4l2_name, value):
            applied.append(f"{v4l2_name}={value}")
        else:
            ok = any(
                _v4l2_set(dev, alt, value) for alt in _ALT_NAMES.get(v4l2_name, ())
            )
            if ok:
                applied.append(f"{v4l2_name}={value}")
            else:
                failed.append(v4l2_name)

    if applied:
        print(f"PS3 Eye [{dev}]: {', '.join(applied)}")
    if failed:
        print(f"PS3 Eye [{dev}]: could not set {', '.join(failed)}")


class WebcamVideoStream:
    """Thread-safe, non-blocking wrapper around cv2.VideoCapture.

    Works with device indices (0, 1, ...) and network stream URLs.
    """

    def __init__(
        self,
        src: Union[int, str] = 0,
        *,
        target_fps: int = 60,
        width: int = 0,
        height: int = 0,
    ) -> None:
        self._cap = cv2.VideoCapture(src)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera/stream: {src}")

        if isinstance(src, int):
            self._cap.set(cv2.CAP_PROP_FPS, target_fps)
            if width > 0:
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            if height > 0:
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            apply_ps3eye_settings(src)

        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._ret = False
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self, warmup_timeout: float = 3.0) -> "WebcamVideoStream":
        """Start the background capture thread and wait for the first frame.

        Blocks up to *warmup_timeout* seconds for the reader thread to
        deliver the first valid frame. Returns self for chaining.
        """
        if self._running:
            return self
        self._running = True
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

        import time as _time
        deadline = _time.time() + warmup_timeout
        while _time.time() < deadline:
            with self._lock:
                if self._ret and self._frame is not None:
                    return self
            _time.sleep(0.05)

        return self

    def _reader(self) -> None:
        while self._running:
            ret, frame = self._cap.read()
            with self._lock:
                self._ret = ret
                self._frame = frame

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Return the latest frame (non-blocking). Same signature as cv2.VideoCapture.read()."""
        with self._lock:
            return self._ret, self._frame.copy() if self._frame is not None else None

    def get(self, prop_id: int) -> float:
        """Proxy for cv2.VideoCapture.get()."""
        return self._cap.get(prop_id)

    def set(self, prop_id: int, value: float) -> bool:
        """Proxy for cv2.VideoCapture.set()."""
        return self._cap.set(prop_id, value)

    def isOpened(self) -> bool:
        return self._cap.isOpened()

    def release(self) -> None:
        """Stop the thread and release the camera."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._cap.release()

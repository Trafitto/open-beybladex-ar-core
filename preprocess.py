"""
Frame preprocessing for improved detection (HSV, CLAHE, saturation).
"""
from typing import Any

import cv2
import numpy as np

_clahe_cache: dict = {"clip": None, "tile": None, "obj": None}


def _get_clahe(clip: float, tile: tuple[int, int]) -> cv2.CLAHE:
    """Return a cached CLAHE object, only recreating when parameters change."""
    if _clahe_cache["clip"] != clip or _clahe_cache["tile"] != tile:
        _clahe_cache["clip"] = clip
        _clahe_cache["tile"] = tile
        _clahe_cache["obj"] = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)
    return _clahe_cache["obj"]


def preprocess_frame_hsv(
    frame: np.ndarray,
    *,
    enabled: bool = True,
    clahe_clip: float = 2.0,
    clahe_tile: tuple[int, int] = (8, 8),
    sat_scale: float = 1.0,
) -> np.ndarray:
    """
    Convert frame to HSV, apply CLAHE on V channel, optionally boost saturation,
    then convert back to BGR.

    All parameters are optional; use for dependency injection in tests.
    """
    if not enabled:
        return frame

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    clahe = _get_clahe(clahe_clip, clahe_tile)
    hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])

    if sat_scale != 1.0:
        s = hsv[:, :, 1].astype(np.float32) * sat_scale
        hsv[:, :, 1] = np.clip(s, 0, 255).astype(np.uint8)

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def preprocess_frame_hsv_from_config(frame: np.ndarray, cfg: Any) -> np.ndarray:
    """
    preprocess_frame_hsv using config module attributes.
    Use this for production; use preprocess_frame_hsv directly in tests.
    """
    return preprocess_frame_hsv(
        frame,
        enabled=getattr(cfg, "HSV_PREPROCESS_ENABLED", False),
        clahe_clip=float(getattr(cfg, "HSV_CLAHE_CLIP", 2.0)),
        clahe_tile=tuple(getattr(cfg, "HSV_CLAHE_TILE", (8, 8))),
        sat_scale=float(getattr(cfg, "HSV_SAT_SCALE", 1.0)),
    )

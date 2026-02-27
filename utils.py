"""
Utility functions: smoothing, distance, and Beyblade identity.
"""
from collections import deque
from typing import Any, List, Optional, Tuple

import numpy as np


def hue_to_color_name(hue: float) -> str:
    """Map OpenCV hue (0-180) to human-readable color name."""
    if hue < 0:
        return "Unknown"
    h = int(hue) % 180
    if h <= 10 or h >= 170:
        return "Red"
    if h <= 25:
        return "Orange"
    if h <= 35:
        return "Yellow"
    if h <= 85:
        return "Green"
    if h <= 100:
        return "Cyan"
    if h <= 130:
        return "Blue"
    return "Purple"


def get_bey_label(bey: Any, slot_index: int, custom_names: Optional[List[str]] = None) -> str:
    """Label for display: custom name if set, else color from hue."""
    if custom_names and slot_index < len(custom_names) and custom_names[slot_index]:
        return custom_names[slot_index]
    hue = getattr(bey, "color_hue_origin", -1)
    if hue < 0:
        hue = getattr(bey, "color_hue", -1)
    return hue_to_color_name(hue)


def moving_average_position(
    history: deque,
    new_point: Tuple[float, float],
    window_size: int,
) -> Tuple[float, float]:
    """
    Smooth position using moving average over last window_size points.
    If window_size <= 1 returns new_point as-is.
    """
    if window_size <= 1:
        return new_point
    history.append(new_point)
    while len(history) > window_size:
        history.popleft()
    arr = np.array(history)
    return float(np.mean(arr[:, 0])), float(np.mean(arr[:, 1]))


def distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Euclidean distance between two points (x, y)."""
    return float(np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2))

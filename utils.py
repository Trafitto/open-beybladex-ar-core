"""
Utility functions: smoothing and helpers.
"""
from collections import deque
from typing import Tuple

import numpy as np


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

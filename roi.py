"""
Interactive arena ROI selection and rail mask point selection.
"""
import json
import os

import cv2
import numpy as np

from utils import distance


def load_rail_mask_points(
    filepath: str,
    *,
    frame_shape: tuple[int, ...] | None = None,
) -> list[tuple[int, int]] | None:
    """Load polygon points from JSON file. Scale if frame_shape differs from saved. Returns None if file missing or invalid."""
    if not filepath or not os.path.isfile(filepath):
        return None
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        pts = data.get("points", data) if isinstance(data, dict) else data
        points = [tuple(p) for p in pts]
        if not frame_shape or len(points) < 3:
            return points
        saved_w = data.get("frame_w")
        saved_h = data.get("frame_h")
        if saved_w is None or saved_h is None or (saved_w, saved_h) == (frame_shape[1], frame_shape[0]):
            return points
        h, w = frame_shape[:2]
        scale_x = w / saved_w
        scale_y = h / saved_h
        return [(int(x * scale_x), int(y * scale_y)) for x, y in points]
    except (json.JSONDecodeError, TypeError, KeyError, ZeroDivisionError):
        return None


def save_rail_mask_points(
    filepath: str,
    points: list[tuple[int, int]],
    *,
    frame_shape: tuple[int, ...] | None = None,
) -> bool:
    """Save polygon points to JSON file. Optionally save frame_shape for scaling on load."""
    if not filepath or len(points) < 3:
        return False
    parent = os.path.dirname(filepath)
    if parent:
        os.makedirs(parent, exist_ok=True)
    data: dict = {"points": [[int(x), int(y)] for x, y in points]}
    if frame_shape:
        data["frame_w"] = frame_shape[1]
        data["frame_h"] = frame_shape[0]
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return True
    except (OSError, TypeError):
        return False


def select_rail_mask_points(frame: np.ndarray) -> list[tuple[int, int]] | None:
    """
    Interactive rail mask: click points along the green rail (in order).
    The rail is not circular; use 8-12 points to trace its shape.
    Press [c] to confirm, [r] to reset last point, [q] to skip.
    Minimum 3 points required for a polygon.
    """
    win = "Select Rail Mask (polygon)"
    display = frame.copy()
    points: list[tuple[int, int]] = []

    def _draw() -> None:
        display[:] = frame
        h, w = frame.shape[:2]
        instr = "Click along inner edge of rail; green = tracking area (inside). [c]onfirm  [r]undo  [q]skip"
        cv2.putText(display, instr, (10, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        cv2.putText(
            display, f"Points: {len(points)}", (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
        )
        for i, pt in enumerate(points):
            cv2.circle(display, pt, 4, (0, 255, 0), -1)
            if i > 0:
                cv2.line(display, points[i - 1], pt, (0, 255, 0), 2)
        if len(points) >= 3:
            pts = np.array(points, dtype=np.int32)
            overlay = display.copy()
            cv2.fillPoly(overlay, [pts], (0, 128, 0))
            cv2.addWeighted(overlay, 0.3, display, 0.7, 0, display)

    def _mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        points.append((x, y))
        _draw()
        cv2.imshow(win, display)

    _draw()
    cv2.imshow(win, display)
    cv2.setMouseCallback(win, _mouse)

    while True:
        key = cv2.waitKey(50) & 0xFF
        if key == ord("c") and len(points) >= 3:
            break
        if key == ord("r") and points:
            points.pop()
            _draw()
            cv2.imshow(win, display)
        if key == ord("q"):
            points = []
            break

    cv2.destroyWindow(win)
    return points if len(points) >= 3 else None


def select_red_zone(frame: np.ndarray) -> tuple[int, int, int] | None:
    """
    Interactive red zone (high-priority) selection.

    Two clicks: center, then edge of circle. Press [c] to confirm, [r] to reset, [q] to skip.
    """
    win = "Select Red Zone (high-priority)"
    display = frame.copy()
    points: list[tuple[int, int]] = []
    result: list[tuple[int, int, int] | None] = [None]

    def _draw() -> None:
        display[:] = frame
        h, w = frame.shape[:2]
        instr = "Click CENTER, then EDGE of red zone. [c]onfirm  [r]eset  [q]skip"
        cv2.putText(display, instr, (10, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        if len(points) >= 1:
            cv2.drawMarker(display, points[0], (0, 0, 255), cv2.MARKER_CROSS, 12, 2)
        if len(points) >= 2:
            r = int(distance(points[0], points[1]))
            cv2.circle(display, points[0], r, (0, 0, 255), 2)
            result[0] = (points[0][0], points[0][1], r)

    def _mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if len(points) < 2:
            points.append((x, y))
        _draw()
        cv2.imshow(win, display)

    _draw()
    cv2.imshow(win, display)
    cv2.setMouseCallback(win, _mouse)

    while True:
        key = cv2.waitKey(50) & 0xFF
        if key == ord("c") and result[0] is not None:
            break
        if key == ord("r"):
            points.clear()
            result[0] = None
            _draw()
            cv2.imshow(win, display)
        if key == ord("q"):
            result[0] = None
            break

    cv2.destroyWindow(win)
    return result[0]


def save_red_zone(
    filepath: str,
    cx: int,
    cy: int,
    r: int,
    *,
    frame_shape: tuple[int, ...] | None = None,
) -> bool:
    """Save red zone (center, radius) to JSON file. Optionally save frame_shape for scaling on load."""
    if not filepath:
        return False
    parent = os.path.dirname(filepath)
    if parent:
        os.makedirs(parent, exist_ok=True)
    data = {"cx": int(cx), "cy": int(cy), "r": int(r)}
    if frame_shape:
        data["frame_w"] = frame_shape[1]
        data["frame_h"] = frame_shape[0]
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return True
    except (OSError, TypeError):
        return False


def load_red_zone(
    filepath: str,
    *,
    frame_shape: tuple[int, ...] | None = None,
) -> tuple[int, int, int] | None:
    """Load red zone from JSON. Scale if frame_shape differs from saved. Returns None if missing/invalid."""
    if not filepath or not os.path.isfile(filepath):
        return None
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        cx = int(data["cx"])
        cy = int(data["cy"])
        r = int(data["r"])
        if not frame_shape:
            return (cx, cy, r)
        saved_w = data.get("frame_w")
        saved_h = data.get("frame_h")
        if saved_w is None or saved_h is None or (saved_w, saved_h) == (frame_shape[1], frame_shape[0]):
            return (cx, cy, r)
        h, w = frame_shape[:2]
        scale_x = w / saved_w
        scale_y = h / saved_h
        return (int(cx * scale_x), int(cy * scale_y), int(r * min(scale_x, scale_y)))
    except (json.JSONDecodeError, TypeError, KeyError, ZeroDivisionError):
        return None


def select_arena_roi(frame: np.ndarray) -> tuple[int, int, int] | tuple[int, int, int, int] | None:
    """
    Interactive arena ROI selection.

    Two circles (dual ROI): click center, inner edge, outer edge.
    Single circle: click center, edge (then press 'c' without third click).
    Press 'c' to confirm, 'r' to reset, 'q' to skip.
    """
    win = "Select Arena ROI"
    display = frame.copy()
    points: list[tuple[int, int]] = []
    result: list[tuple[int, int, int] | tuple[int, int, int, int] | None] = [None]

    def _draw() -> None:
        display[:] = frame
        h, w = frame.shape[:2]
        instr = (
            "Click CENTER, then INNER edge, then OUTER edge (or 2 clicks for single).  [c]onfirm  [r]eset  [q]skip"
        )
        cv2.putText(display, instr, (10, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        if len(points) >= 1:
            cv2.drawMarker(display, points[0], (0, 255, 0), cv2.MARKER_CROSS, 12, 2)
        if len(points) >= 2:
            r_inner = int(distance(points[0], points[1]))
            cv2.circle(display, points[0], r_inner, (0, 0, 255), 2)
            result[0] = (points[0][0], points[0][1], r_inner)
        if len(points) >= 3:
            r_outer = int(distance(points[0], points[2]))
            cv2.circle(display, points[0], r_inner, (0, 0, 255), 2)
            cv2.circle(display, points[0], r_outer, (0, 255, 0), 2)
            result[0] = (points[0][0], points[0][1], r_inner, r_outer)

    def _mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if len(points) < 3:
            points.append((x, y))
        _draw()
        cv2.imshow(win, display)

    _draw()
    cv2.imshow(win, display)
    cv2.setMouseCallback(win, _mouse)

    while True:
        key = cv2.waitKey(50) & 0xFF
        if key == ord("c") and result[0] is not None:
            break
        if key == ord("r"):
            points.clear()
            result[0] = None
            _draw()
            cv2.imshow(win, display)
        if key == ord("q"):
            result[0] = None
            break

    cv2.destroyWindow(win)
    return result[0]

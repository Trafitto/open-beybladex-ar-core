"""
Interactive arena ROI selection.
"""
import cv2
import numpy as np

from utils import distance


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

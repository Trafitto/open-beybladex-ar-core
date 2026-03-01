"""
Debug overlay: tracking state, colour masks, ROI circles.
"""
from typing import Any, Protocol

import cv2
import numpy as np

import config
from utils import get_bey_label


class TrackerLike(Protocol):
    def get_states(self): ...
    def get_last_circle_count(self) -> int: ...
    def is_bootstrapped(self) -> bool: ...
    def get_arena_roi_low(self): ...
    def get_arena_roi_high(self): ...
    def get_arena_roi(self): ...
    def get_rim_circle(self): ...
    def get_rail_mask(self): ...
    def get_polygon_points(self): ...
    def get_debug_masks(self, frame: np.ndarray): ...


class CollisionDetectorLike(Protocol):
    event_count: int


def draw_debug_overlay(
    frame: np.ndarray,
    tracker: TrackerLike,
    collision_detector: CollisionDetectorLike,
    frame_index: int,
    *,
    max_bey_count: int = 2,
    color_bey_1: tuple[int, int, int] = (0, 255, 0),
    color_bey_2: tuple[int, int, int] = (255, 0, 0),
    collision_margin_px: int = 2,
) -> None:
    """Show colour tracking state, per-bey colour masks, and collision margin zones for -d/--debug."""
    h, w = frame.shape[:2]
    n_tracked = len(tracker.get_states())
    mode = (
        f"tracking {n_tracked}/{max_bey_count}"
        if tracker.is_bootstrapped()
        else "scanning..."
    )
    margin = max(collision_margin_px, 1)
    lines = [
        f"mode: {mode}",
        f"detections: {tracker.get_last_circle_count()}",
        f"frame {w}x{h} #{frame_index}",
        f"collisions: {collision_detector.event_count}",
        f"collision margin: {margin}px (magenta=zone, gray=bey)",
    ]
    sorted_states = sorted(tracker.get_states(), key=lambda b: b.id)
    identities = getattr(config, "BEY_IDENTITIES", None) or []
    for slot, b in enumerate(sorted_states):
        identity = get_bey_label(b, slot, identities)
        hue_label = f"hue={b.color_hue:.0f}" if b.color_hue >= 0 else "hue=?"
        lines.append(f"  {identity} (bey#{b.id}) {hue_label} unseen={b.frames_since_seen}")

    y_pos = 24
    for line in lines:
        cv2.putText(frame, line, (8, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
        cv2.putText(frame, line, (8, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        y_pos += 18

    swatch_x = 8
    overlay_colors = [color_bey_1, color_bey_2]
    for slot, b in enumerate(sorted_states):
        if b.color_hue < 0:
            continue
        identity = get_bey_label(b, slot, identities)
        swatch_hsv = np.array([[[int(b.color_hue), 255, 255]]], dtype=np.uint8)
        swatch_bgr = cv2.cvtColor(swatch_hsv, cv2.COLOR_HSV2BGR)
        sc = tuple(int(v) for v in swatch_bgr[0, 0])
        cv2.rectangle(frame, (swatch_x, y_pos), (swatch_x + 14, y_pos + 14), sc, -1)
        cv2.rectangle(frame, (swatch_x, y_pos), (swatch_x + 14, y_pos + 14), (255, 255, 255), 1)
        cv2.putText(
            frame,
            identity,
            (swatch_x + 20, y_pos + 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
        )
        swatch_x += 80

    polygon = tracker.get_polygon_points()
    if polygon:
        pts = np.array(polygon, dtype=np.int32)
        cv2.polylines(frame, [pts], True, (0, 255, 0), 1)
    roi_low = tracker.get_arena_roi_low()
    roi_high = tracker.get_arena_roi_high()
    if roi_low:
        cv2.circle(frame, (roi_low[0], roi_low[1]), roi_low[2], (0, 255, 0), 1)
    if roi_high and not getattr(config, "DEBUG_HIDE_RED_CIRCLE_WHEN_POLYGON", False):
        cv2.circle(frame, (roi_high[0], roi_high[1]), roi_high[2], (0, 0, 255), 1)
    if not polygon and not roi_low and not roi_high:
        roi = tracker.get_arena_roi()
        if roi:
            cv2.circle(frame, (roi[0], roi[1]), roi[2], (0, 255, 255), 1)
    rim = tracker.get_rim_circle()
    if rim:
        cv2.circle(frame, (rim[0], rim[1]), rim[2], (255, 200, 0), 1)

    rail_mask = tracker.get_rail_mask()
    if rail_mask is not None and tracker.get_polygon_points() is None:
        overlay = frame.copy()
        overlay[rail_mask > 0] = (0, 128, 0)
        cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)

    hand_mask = getattr(tracker, "get_hand_mask", lambda: None)()
    if hand_mask is not None:
        overlay = frame.copy()
        overlay[hand_mask > 0] = (0, 165, 255)
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

    for b in tracker.get_states():
        cx, cy = int(b.position[0]), int(b.position[1])
        r_bey = int(b.radius)
        r_zone = int(b.radius + margin)
        cv2.circle(frame, (cx, cy), r_bey, (180, 180, 180), 1)
        cv2.circle(frame, (cx, cy), r_zone, (255, 0, 255), 1)

    masks = tracker.get_debug_masks(frame)
    if masks:
        overlay = frame.copy()
        for bey_id, mask in masks.items():
            oc = overlay_colors[bey_id % 2]
            overlay[mask > 0] = oc
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)


def draw_debug_overlay_from_config(
    frame: np.ndarray,
    tracker: TrackerLike,
    collision_detector: CollisionDetectorLike,
    frame_index: int,
    cfg: Any,
) -> None:
    """draw_debug_overlay using config module attributes."""
    margin = max(getattr(cfg, "BEY_COLLISION_MARGIN_PX", 2), 1)
    draw_debug_overlay(
        frame,
        tracker,
        collision_detector,
        frame_index,
        max_bey_count=getattr(cfg, "MAX_BEY_COUNT", 2),
        color_bey_1=getattr(cfg, "COLOR_BEY_1", (0, 255, 0)),
        color_bey_2=getattr(cfg, "COLOR_BEY_2", (255, 0, 0)),
        collision_margin_px=margin,
    )

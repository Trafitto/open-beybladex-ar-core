"""
Visual effects: trails, impact flash, wall hit.
"""
from collections import deque
from typing import Any

import cv2
import numpy as np


def draw_trail_effect(
    frame: np.ndarray,
    states: list,
    trail_history: dict[int, deque],
    *,
    colors: list[tuple[int, int, int]] | None = None,
    trail_max_len: int = 100,
    trail_base_radius: float = 6,
    points_per_segment: int = 1,
) -> None:
    """Draw fading trail circles under each bey."""
    if colors is None:
        colors = [(0, 255, 0), (255, 0, 0)]
    for b in states:
        if b.id not in trail_history:
            trail_history[b.id] = deque(maxlen=trail_max_len)
        hist = trail_history[b.id]
        points = list(hist)
        color = colors[b.id % 2]
        for i, (px, py) in enumerate(points):
            t = (i + 1) / max(1, len(points))
            r = max(1, int(trail_base_radius * t))
            c = (int(color[0] * t), int(color[1] * t), int(color[2] * t))
            cv2.circle(frame, (int(px), int(py)), r, c, -1)
        bx, by = b.position[0], b.position[1]
        if points_per_segment > 1 and len(hist) > 0:
            lx, ly = hist[-1][0], hist[-1][1]
            for k in range(1, points_per_segment):
                s = k / points_per_segment
                hist.append((lx + s * (bx - lx), ly + s * (by - ly)))
        hist.append((bx, by))


def draw_impact_effect(
    frame: np.ndarray,
    center: tuple[float, float],
    frames_left: int,
    impact_force: float = 0.0,
    *,
    total_frames: int = 8,
    max_radius: int = 55,
    force_radius_scale: float = 0.002,
    min_force_baseline: float = 100.0,
    ripple_rings: int = 3,
    color: tuple[int, int, int] = (0, 200, 255),
) -> None:
    """Draw expanding ring(s) at impact point."""
    if total_frames <= 0 or frames_left <= 0:
        return
    progress = 1.0 - (frames_left - 1) / total_frames
    force_factor = 1.0 + max(0.0, impact_force - min_force_baseline) * force_radius_scale
    max_r = int(max_radius * min(force_factor, 1.8))

    cx, cy = int(center[0]), int(center[1])
    alpha = 1.0 - progress * 0.7

    ring_delay = 0.22
    for i in range(ripple_rings):
        ring_progress = max(0.0, progress - i * ring_delay)
        r = int(ring_progress * max_r)
        if r < 1:
            continue
        ring_alpha = alpha * (1.0 - i * 0.25)
        thickness = max(1, int(4 * ring_alpha))
        cv2.circle(frame, (cx, cy), r, color, thickness)

    r = int(progress * max_r)
    if r > 8:
        cv2.circle(frame, (cx, cy), max(1, r - 6), color, max(1, int(3 * alpha)))


def draw_impact_label(
    frame: np.ndarray,
    center: tuple[float, float],
    frames_left: int,
    impact_force: float = 0.0,
    *,
    total_frames: int = 8,
    enabled: bool = True,
    show_force: bool = False,
    color: tuple[int, int, int] = (0, 200, 255),
) -> None:
    """Draw 'IMPACT!' text at impact point."""
    if total_frames <= 0 or frames_left <= 0:
        return
    if not enabled:
        return
    cx, cy = int(center[0]), int(center[1])
    text = "IMPACT!"
    if show_force and impact_force > 0:
        text = f"IMPACT! {int(impact_force)}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    tx = cx - tw // 2
    ty = cy + th // 2
    cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)


def draw_wall_hit(
    frame: np.ndarray,
    bey,
    rim_circle: tuple[int, int, int],
    frames_left: int,
    *,
    total_frames: int = 6,
    color: tuple[int, int, int] = (255, 180, 0),
) -> None:
    """Draw arc on rim at wall hit location."""
    if total_frames <= 0 or frames_left <= 0:
        return
    rcx, rcy, rr = rim_circle
    bx, by = int(bey.position[0]), int(bey.position[1])
    angle = np.degrees(np.arctan2(by - rcy, bx - rcx))
    arc_span = 25
    axes = (rr, rr)
    cv2.ellipse(
        frame, (rcx, rcy), axes,
        0, angle - arc_span, angle + arc_span,
        color, 1,
    )


def draw_trail_effect_from_config(
    frame: np.ndarray, states: list, trail_history: dict[int, deque], cfg: Any
) -> None:
    """draw_trail_effect using config module attributes."""
    draw_trail_effect(
        frame,
        states,
        trail_history,
        colors=[getattr(cfg, "COLOR_BEY_1", (0, 255, 0)), getattr(cfg, "COLOR_BEY_2", (255, 0, 0))],
        trail_max_len=getattr(cfg, "TRAIL_MAX_LEN", 100),
        trail_base_radius=getattr(cfg, "TRAIL_BASE_RADIUS", 6),
        points_per_segment=getattr(cfg, "TRAIL_POINTS_PER_SEGMENT", 1),
    )


def draw_impact_effect_from_config(
    frame: np.ndarray,
    center: tuple[float, float],
    frames_left: int,
    impact_force: float,
    cfg: Any,
) -> None:
    """draw_impact_effect using config module attributes."""
    draw_impact_effect(
        frame,
        center,
        frames_left,
        impact_force,
        total_frames=getattr(cfg, "IMPACT_FLASH_FRAMES", 8),
        max_radius=getattr(cfg, "IMPACT_FLASH_MAX_RADIUS", 55),
        force_radius_scale=getattr(cfg, "IMPACT_FORCE_RADIUS_SCALE", 0.002),
        min_force_baseline=getattr(cfg, "IMPACT_MIN_FORCE_BASELINE", 100.0),
        ripple_rings=getattr(cfg, "IMPACT_RIPPLE_RINGS", 3),
        color=getattr(cfg, "IMPACT_FLASH_COLOR", (0, 200, 255)),
    )


def draw_impact_label_from_config(
    frame: np.ndarray,
    center: tuple[float, float],
    frames_left: int,
    impact_force: float,
    cfg: Any,
) -> None:
    """draw_impact_label using config module attributes."""
    draw_impact_label(
        frame,
        center,
        frames_left,
        impact_force,
        total_frames=getattr(cfg, "IMPACT_FLASH_FRAMES", 8),
        enabled=getattr(cfg, "IMPACT_LABEL_ENABLED", True),
        show_force=getattr(cfg, "IMPACT_LABEL_SHOW_FORCE", False),
        color=getattr(cfg, "IMPACT_FLASH_COLOR", (0, 200, 255)),
    )


def draw_wall_hit_from_config(
    frame: np.ndarray, bey, rim_circle: tuple[int, int, int], frames_left: int, cfg: Any
) -> None:
    """draw_wall_hit using config module attributes."""
    draw_wall_hit(
        frame,
        bey,
        rim_circle,
        frames_left,
        total_frames=getattr(cfg, "WALL_FLASH_FRAMES", 6),
        color=getattr(cfg, "WALL_FLASH_COLOR", (255, 180, 0)),
    )

"""
Main overlay: bey circles, velocity arrows, IMPACT label.
"""
from typing import Any

import cv2


def draw_overlay(
    frame,
    states: list,
    collision: bool,
    impact_center: tuple[float, float] | None = None,
    *,
    colors: list[tuple[int, int, int]] | None = None,
    radius_scale: float = 1.4,
    velocity_color: tuple[int, int, int] = (0, 255, 255),
    impact_color: tuple[int, int, int] = (0, 0, 255),
    font_scale: float = 0.6,
    font_thickness: int = 2,
) -> None:
    """Draw bey circles, velocity arrows, optional IMPACT label."""
    if colors is None:
        colors = [(0, 255, 0), (255, 0, 0)]
    for b in states:
        color = colors[b.id % 2]
        cx, cy = int(b.position[0]), int(b.position[1])
        r = int(b.radius * radius_scale)
        cv2.circle(frame, (cx, cy), r, color, 2)
        cv2.circle(frame, (cx, cy), 3, color, -1)
        vx, vy = b.velocity
        scale = 0.1
        end = (int(cx + vx * scale), int(cy + vy * scale))
        cv2.arrowedLine(frame, (cx, cy), end, velocity_color, 2)
        label = f"#{b.id} {b.speed:.0f}"
        cv2.putText(
            frame,
            label,
            (cx - 30, cy - r - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            font_thickness,
        )
    if collision:
        h, w = frame.shape[:2]
        if impact_center is not None:
            cx, cy = int(impact_center[0]), int(impact_center[1])
            (tw, th), _ = cv2.getTextSize("IMPACT!", cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
            tx, ty = cx - tw // 2, cy + th // 2
            tx = max(0, min(tx, w - tw))
            ty = max(th, min(ty, h))
            cv2.putText(frame, "IMPACT!", (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 1.5, impact_color, 3)
        else:
            cv2.putText(
                frame, "IMPACT!", (w // 2 - 80, h // 2 - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, impact_color, 3
            )


def draw_overlay_from_config(
    frame,
    states: list,
    collision: bool,
    impact_center: tuple[float, float] | None,
    cfg: Any,
) -> None:
    """draw_overlay using config module attributes."""
    draw_overlay(
        frame,
        states,
        collision,
        impact_center,
        colors=[getattr(cfg, "COLOR_BEY_1", (0, 255, 0)), getattr(cfg, "COLOR_BEY_2", (255, 0, 0))],
        radius_scale=getattr(cfg, "BEY_RADIUS_SCALE", 1.4),
        velocity_color=getattr(cfg, "COLOR_VELOCITY", (0, 255, 255)),
        impact_color=getattr(cfg, "COLOR_IMPACT", (0, 0, 255)),
        font_scale=getattr(cfg, "FONT_SCALE", 0.6),
        font_thickness=getattr(cfg, "FONT_THICKNESS", 2),
    )

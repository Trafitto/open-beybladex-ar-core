"""
Physics: velocity computation and collision detection with debouncing.
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

import config
from utils import distance


def compute_velocity(
    current: Tuple[float, float],
    previous: Tuple[float, float],
    dt: float,
) -> Tuple[Tuple[float, float], float]:
    if dt <= 0:
        return (0.0, 0.0), 0.0
    vx = (current[0] - previous[0]) / dt
    vy = (current[1] - previous[1]) / dt
    scalar = float(np.sqrt(vx * vx + vy * vy))
    return (vx, vy), scalar


def check_collision(
    center1: Tuple[float, float],
    r1: float,
    center2: Tuple[float, float],
    r2: float,
) -> bool:
    return distance(center1, center2) < (r1 + r2)


def compute_closing_speed(
    pos0: Tuple[float, float],
    pos1: Tuple[float, float],
    vel0: Tuple[float, float],
    vel1: Tuple[float, float],
) -> float:
    """Closing speed: projection of relative velocity onto the center-to-center
    axis.  Positive means the beys are approaching; negative means separating."""
    dx = pos1[0] - pos0[0]
    dy = pos1[1] - pos0[1]
    d = np.sqrt(dx * dx + dy * dy)
    if d < 1e-6:
        return 0.0
    ux, uy = dx / d, dy / d
    rvx = vel0[0] - vel1[0]
    rvy = vel0[1] - vel1[1]
    return float(rvx * ux + rvy * uy)


def _velocity_direction_changed(
    v_obs: Tuple[float, float],
    v_pred: Tuple[float, float],
    min_speed: float = 5.0,
    cos_threshold: float = 0.5,
) -> bool:
    """True if observed velocity deflects from predicted (reflection/collision)."""
    mag_o = float(np.sqrt(v_obs[0] ** 2 + v_obs[1] ** 2))
    mag_p = float(np.sqrt(v_pred[0] ** 2 + v_pred[1] ** 2))
    if mag_o < min_speed or mag_p < min_speed:
        return False
    dot = v_obs[0] * v_pred[0] + v_obs[1] * v_pred[1]
    cos_angle = dot / (mag_o * mag_p)
    return cos_angle < cos_threshold


def has_velocity_reversal(
    velocities: List[Tuple[float, float]],
    predicted_velocities: List[Optional[Tuple[float, float]]],
    min_speed: float = 5.0,
    cos_threshold: float = 0.5,
) -> bool:
    """True if at least one bey shows direction change vs Kalman prediction."""
    for v_obs, v_pred in zip(velocities, predicted_velocities):
        if v_pred is None:
            continue
        if _velocity_direction_changed(v_obs, v_pred, min_speed, cos_threshold):
            return True
    return False


def check_wall_collision(
    bey_center: Tuple[float, float],
    bey_radius: float,
    rim_center: Tuple[float, float],
    rim_radius: float,
    margin: float = 0.0,
) -> bool:
    d = distance(bey_center, rim_center)
    return (d + bey_radius) >= (rim_radius - margin)


def detect_wall_collisions(
    positions: List[Tuple[float, float]],
    radii: List[float],
    rim_circle: Tuple[float, float, float],
    margin: float = 0.0,
) -> List[int]:
    rcx, rcy, rr = rim_circle
    hits: List[int] = []
    for i, (pos, r) in enumerate(zip(positions, radii)):
        if check_wall_collision(pos, r, (rcx, rcy), rr, margin):
            hits.append(i)
    return hits


@dataclass
class CollisionEvent:
    """Represents a single collision between two beyblades."""
    bey_id_a: int
    bey_id_b: int
    position: Tuple[float, float]
    relative_speed: float
    impact_force: float
    frame: int
    closing_speed: float = 0.0


class CollisionDetector:
    """Detects bey-bey collisions with multi-layer validation.

    Filters:
      1. Circle overlap (geometry)
      2. Leading-edge debounce + cooldown
      3. Closing speed (beys must be approaching, not passing by)
      4. Minimum overlap depth (rejects edge-grazing from tilt wobble)
      5. Minimum relative speed
      6. Optional Kalman velocity-reversal confirmation
      7. Radius stability (rejects when detected radius jumps -- tilt artifact)
    """

    def __init__(self) -> None:
        self._cooldown_remaining: int = 0
        self._last_overlapping: bool = False
        self._events: List[CollisionEvent] = []
        self._frame: int = 0
        self._prev_radii: List[float] = []

    def update(
        self,
        positions: List[Tuple[float, float]],
        radii: List[float],
        velocities: List[Tuple[float, float]],
        frame_index: int,
        predicted_velocities: Optional[List[Optional[Tuple[float, float]]]] = None,
        detected_radii: Optional[List[float]] = None,
    ) -> Optional[CollisionEvent]:
        """Check for a collision this frame.

        *detected_radii*: the tracker's raw detected radius for each bey
        (distinct from the fixed collision radii).  Used to reject tilt
        artifacts where the apparent radius spikes.
        """
        self._frame = frame_index

        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1

        if len(positions) < 2:
            self._last_overlapping = False
            return None

        overlapping = check_collision(
            positions[0], radii[0],
            positions[1], radii[1],
        )

        if not overlapping:
            self._last_overlapping = False
            if detected_radii and len(detected_radii) >= 2:
                self._prev_radii = list(detected_radii[:2])
            return None

        if self._last_overlapping or self._cooldown_remaining > 0:
            self._last_overlapping = overlapping
            return None

        # -- Filter: closing speed (must be approaching) --
        v0 = velocities[0] if len(velocities) > 0 else (0.0, 0.0)
        v1 = velocities[1] if len(velocities) > 1 else (0.0, 0.0)
        closing = compute_closing_speed(positions[0], positions[1], v0, v1)
        min_closing = float(getattr(
            config, "COLLISION_MIN_CLOSING_SPEED", 15.0
        ))
        if closing < min_closing:
            self._last_overlapping = overlapping
            return None

        # -- Filter: minimum overlap depth --
        d = distance(positions[0], positions[1])
        overlap_depth = (radii[0] + radii[1]) - d
        min_overlap = float(getattr(config, "COLLISION_MIN_OVERLAP_PX", 3.0))
        if overlap_depth < min_overlap:
            self._last_overlapping = overlapping
            return None

        # -- Filter: radius stability (tilt detection) --
        max_radius_jump = float(getattr(
            config, "COLLISION_MAX_RADIUS_JUMP", 0.5
        ))
        if (detected_radii and len(detected_radii) >= 2
                and self._prev_radii and len(self._prev_radii) >= 2):
            for i in range(2):
                prev_r = self._prev_radii[i]
                curr_r = detected_radii[i]
                if prev_r > 0:
                    ratio = abs(curr_r - prev_r) / prev_r
                    if ratio > max_radius_jump:
                        self._last_overlapping = overlapping
                        if detected_radii:
                            self._prev_radii = list(detected_radii[:2])
                        return None

        # -- Filter: Kalman velocity reversal --
        use_kalman_confirm = getattr(config, "COLLISION_KALMAN_CONFIRM", False)
        if use_kalman_confirm and predicted_velocities and len(predicted_velocities) >= 2:
            preds = predicted_velocities[:2]
            min_speed = getattr(config, "COLLISION_VELOCITY_REVERSAL_MIN_SPEED", 5.0)
            cos_thresh = getattr(config, "COLLISION_VELOCITY_REVERSAL_COS", 0.5)
            if not has_velocity_reversal(velocities[:2], preds, min_speed, cos_thresh):
                self._last_overlapping = overlapping
                return None

        # -- Filter: minimum relative speed --
        rel_vx = v0[0] - v1[0]
        rel_vy = v0[1] - v1[1]
        relative_speed = float(np.sqrt(rel_vx ** 2 + rel_vy ** 2))
        min_approach = float(getattr(config, "COLLISION_MIN_APPROACH_SPEED", 40.0))
        if relative_speed < min_approach:
            self._last_overlapping = overlapping
            return None

        # -- Collision confirmed --
        self._last_overlapping = True
        self._cooldown_remaining = int(getattr(
            config, "COLLISION_COOLDOWN_FRAMES", 15
        ))
        if detected_radii and len(detected_radii) >= 2:
            self._prev_radii = list(detected_radii[:2])

        impact_x = (positions[0][0] + positions[1][0]) / 2.0
        impact_y = (positions[0][1] + positions[1][1]) / 2.0
        impact_force = relative_speed * max(overlap_depth, 1.0)

        event = CollisionEvent(
            bey_id_a=0,
            bey_id_b=1,
            position=(impact_x, impact_y),
            relative_speed=relative_speed,
            impact_force=impact_force,
            frame=frame_index,
            closing_speed=closing,
        )
        self._events.append(event)
        return event

    @property
    def is_overlapping(self) -> bool:
        return self._last_overlapping

    @property
    def events(self) -> List[CollisionEvent]:
        return list(self._events)

    @property
    def event_count(self) -> int:
        return len(self._events)

"""
Beyblade tracker: Hough circles + center color + frame differencing + Kalman.

Every frame:
    1. Hough circles finds all circular objects in the arena.
    2. Each circle's center chip colour is sampled (median hue of saturated px).
    3. Circles are matched to tracked beys by proximity + hue similarity.
    4. Kalman filter smooths and predicts each bey's position.
    5. Unmatched circles that show local motion (frame differencing) and have
       a unique colour become new tracked beys.
"""
import os
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

import config
from physics import compute_velocity
from utils import distance, moving_average_position


def _fit_circle_lsq(pts: np.ndarray) -> Tuple[float, float, float]:
    """Algebraic least-squares circle fit (Kasa method)."""
    if len(pts) < 3:
        return 0.0, 0.0, -1.0
    x = pts[:, 0]
    y = pts[:, 1]
    A = np.column_stack([x, y, np.ones(len(x))])
    b = x ** 2 + y ** 2
    try:
        result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return 0.0, 0.0, -1.0
    cx = result[0] / 2.0
    cy = result[1] / 2.0
    r = np.sqrt(result[2] + cx ** 2 + cy ** 2)
    return float(cx), float(cy), float(r)


def _make_kalman_filter() -> cv2.KalmanFilter:
    """4-state (x, y, vx, vy) / 2-measurement (x, y) Kalman filter."""
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0]], np.float32
    )
    kf.transitionMatrix = np.array(
        [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32
    )
    q = float(getattr(config, "KALMAN_PROCESS_NOISE", 1e-2))
    r = float(getattr(config, "KALMAN_MEASUREMENT_NOISE", 5e-1))
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * q
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * r
    kf.errorCovPost = np.eye(4, dtype=np.float32)
    return kf


def _hue_distance(h1: float, h2: float) -> float:
    """Shortest angular distance between two OpenCV hues (0-180)."""
    d = abs(h1 - h2)
    return min(d, 180.0 - d)


def _clamp_velocity_to_rim(
    position: Tuple[float, float],
    velocity: Tuple[float, float],
    stadium: Tuple[int, int, int],
    inner_frac: float = 0.92,
) -> Tuple[float, float]:
    """Remove outward radial velocity when bey is near the stadium rim.

    When the bey is in the outer band (distance >= inner_frac * radius from center),
    project velocity onto the tangent so it cannot point outside the circle.
    """
    scx, scy, sr = stadium
    dx = position[0] - scx
    dy = position[1] - scy
    dist = np.sqrt(dx * dx + dy * dy)
    if dist < 1:
        return velocity
    inner_r = sr * inner_frac
    if dist < inner_r:
        return velocity
    r_hat_x = dx / dist
    r_hat_y = dy / dist
    v_radial = velocity[0] * r_hat_x + velocity[1] * r_hat_y
    if v_radial <= 0:
        return velocity
    vx = velocity[0] - v_radial * r_hat_x
    vy = velocity[1] - v_radial * r_hat_y
    return (float(vx), float(vy))


def _predict_circular(
    positions: List[Tuple[float, float]],
    min_points: int = 5,
    max_fit_error_frac: float = 0.15,
) -> Optional[Tuple[float, float]]:
    """Predict next position along circular arc from recent positions.

    Fits circle to positions, estimates angular velocity, predicts next point.
    Returns None if fit fails or orbit is not detected.
    """
    if len(positions) < min_points:
        return None
    pts = np.array(positions, dtype=np.float64)
    cx, cy, r = _fit_circle_lsq(pts)
    if r <= 0 or r > 500:
        return None
    dx = pts[:, 0] - cx
    dy = pts[:, 1] - cy
    angles = np.arctan2(dy, dx)
    angles = np.unwrap(angles)
    fit_radii = np.sqrt(dx * dx + dy * dy)
    residual = np.mean(np.abs(fit_radii - r)) / max(r, 1)
    if residual > max_fit_error_frac:
        return None
    ang_vel = (angles[-1] - angles[0]) / max(len(angles) - 1, 1)
    next_angle = angles[-1] + ang_vel
    px = cx + r * np.cos(next_angle)
    py = cy + r * np.sin(next_angle)
    return (float(px), float(py))


# ------------------------------------------------------------------ #
#  BeyState                                                            #
# ------------------------------------------------------------------ #


@dataclass
class BeyState:
    """State for a single tracked bey."""

    id: int
    position: Tuple[float, float]
    previous_position: Tuple[float, float]
    velocity: Tuple[float, float]
    speed: float
    radius: float
    color_hue: float = -1.0
    color_hue_origin: float = -1.0
    radius_origin: float = -1.0
    position_history: deque = field(
        default_factory=lambda: deque(maxlen=config.SMOOTH_WINDOW_SIZE + 2)
    )
    circular_history: deque = field(
        default_factory=lambda: deque(
            maxlen=getattr(config, "CIRCULAR_HISTORY_LEN", 12)
        ),
        compare=False,
        repr=False,
    )
    frames_stationary: int = 0
    frames_since_seen: int = 0
    _kalman: Any = field(default=None, compare=False, repr=False)
    predicted_velocity: Optional[Tuple[float, float]] = None

    def _ensure_kalman(self) -> None:
        if self._kalman is None and getattr(config, "KALMAN_ENABLED", False):
            self._kalman = _make_kalman_filter()
            self._kalman.statePre = np.array(
                [self.position[0], self.position[1], 0.0, 0.0], np.float32
            ).reshape(4, 1)
            self._kalman.statePost = self._kalman.statePre.copy()

    def kalman_predict(self) -> Tuple[float, float]:
        px: float
        py: float
        if getattr(config, "CIRCULAR_PREDICTION_ENABLED", False):
            circ_pred = _predict_circular(list(self.circular_history))
            if circ_pred is not None:
                px, py = circ_pred
                self._ensure_kalman()
                if self._kalman is not None:
                    self._kalman.statePre[0] = np.float32(px)
                    self._kalman.statePre[1] = np.float32(py)
                    vx = px - self.position[0]
                    vy = py - self.position[1]
                    self._kalman.statePre[2] = np.float32(vx)
                    self._kalman.statePre[3] = np.float32(vy)
            else:
                px, py = self._linear_predict()
        else:
            px, py = self._linear_predict()

        max_drift = float(getattr(config, "KALMAN_MAX_PREDICTION_DRIFT", 60))
        dx = px - self.position[0]
        dy = py - self.position[1]
        d = np.sqrt(dx * dx + dy * dy)
        if d > max_drift:
            scale = max_drift / d
            px = self.position[0] + dx * scale
            py = self.position[1] + dy * scale

        if self._kalman is not None:
            self.predicted_velocity = (
                float(self._kalman.statePre[2]),
                float(self._kalman.statePre[3]),
            )
        else:
            self.predicted_velocity = None

        return (px, py)

    def _linear_predict(self) -> Tuple[float, float]:
        self._ensure_kalman()
        if self._kalman is None:
            return (
                self.position[0] + self.velocity[0],
                self.position[1] + self.velocity[1],
            )
        pred = self._kalman.predict()
        return float(pred[0]), float(pred[1])

    def update_from_raw(
        self,
        raw_center: Tuple[float, float],
        raw_radius: float,
        dt: float,
    ) -> None:
        self._ensure_kalman()
        if self._kalman is not None:
            meas = np.array([[raw_center[0]], [raw_center[1]]], np.float32)
            corrected = self._kalman.correct(meas)
            raw_center = (float(corrected[0]), float(corrected[1]))
        smoothed = moving_average_position(
            self.position_history,
            raw_center,
            config.SMOOTH_WINDOW_SIZE,
        )
        self.previous_position = self.position
        self.position = smoothed
        self.radius = raw_radius
        self.velocity, self.speed = compute_velocity(
            self.position, self.previous_position, dt
        )
        self.frames_since_seen = 0
        if getattr(config, "CIRCULAR_PREDICTION_ENABLED", False):
            self.circular_history.append(self.position)
        min_speed = getattr(config, "BOOTSTRAP_MIN_SPEED", 5.0)
        if self.speed >= min_speed:
            self.frames_stationary = 0
        else:
            self.frames_stationary += 1

    def reset_kalman_at(self, position: Tuple[float, float]) -> None:
        self._kalman = None
        self.position = position
        self.previous_position = position
        self.velocity = (0.0, 0.0)
        self.speed = 0.0
        self.predicted_velocity = None
        self.frames_stationary = 0
        self.frames_since_seen = 0
        self.position_history.clear()
        self.circular_history.clear()
        self._ensure_kalman()

    def keep_previous_position(self) -> None:
        self.frames_stationary += 1
        self.frames_since_seen += 1
        self._ensure_kalman()
        if self._kalman is not None:
            self._kalman.statePost = self._kalman.statePre.copy()
            self._kalman.errorCovPost = self._kalman.errorCovPre.copy()
            decay = float(getattr(config, "KALMAN_LOSS_VELOCITY_DECAY", 0.3))
            self._kalman.statePost[2] *= decay
            self._kalman.statePost[3] *= decay
            pred_x = float(self._kalman.statePost[0])
            pred_y = float(self._kalman.statePost[1])
            vel_x = float(self._kalman.statePost[2])
            vel_y = float(self._kalman.statePost[3])
            pred_speed = np.sqrt(vel_x * vel_x + vel_y * vel_y)

            max_vel = float(getattr(config, "KALMAN_MAX_VELOCITY_PX", 0))
            if max_vel > 0 and pred_speed > max_vel:
                self._kalman.statePost[0] = np.float32(self.position[0])
                self._kalman.statePost[1] = np.float32(self.position[1])
                self._kalman.statePost[2] = np.float32(0.0)
                self._kalman.statePost[3] = np.float32(0.0)
                self.previous_position = self.position
                self.velocity = (0.0, 0.0)
                self.speed = 0.0
                return

            max_drift = float(getattr(config, "KALMAN_MAX_PREDICTION_DRIFT", 60))
            dx = pred_x - self.position[0]
            dy = pred_y - self.position[1]
            d = np.sqrt(dx * dx + dy * dy)
            if d > max_drift:
                scale = max_drift / d
                pred_x = self.position[0] + dx * scale
                pred_y = self.position[1] + dy * scale
                self._kalman.statePost[0] = np.float32(pred_x)
                self._kalman.statePost[1] = np.float32(pred_y)

            self.previous_position = self.position
            self.position = (pred_x, pred_y)
            vel = (self.velocity[0] * decay, self.velocity[1] * decay)
            self.velocity = vel
            self.speed *= decay
        else:
            self.previous_position = self.position
            self.velocity = (0.0, 0.0)
            self.speed = 0.0


# ------------------------------------------------------------------ #
#  BeyTracker                                                          #
# ------------------------------------------------------------------ #


class BeyTracker:
    """
    Hough circle tracker for up to MAX_BEY_COUNT beyblades.

    Every frame:
        1. Hough circles -> list of (center, radius)
        2. Sample center chip hue for each circle
        3. Match circles to tracked beys (distance + hue score)
        4. Update matched beys via Kalman correct
        5. Extrapolate unmatched beys via Kalman predict
        6. Register new beys from unmatched moving circles
    """

    def __init__(self) -> None:
        self._bey: List[BeyState] = []
        self._next_id = 0
        self._last_circle_count = 0
        self._frame_index = 0
        self._bootstrapped = False
        self._frames_all_stuck = 0
        self._arena_roi: Optional[Tuple[int, int, int]] = None
        self._arena_roi_high: Optional[Tuple[int, int, int]] = None
        self._arena_roi_low: Optional[Tuple[int, int, int]] = None
        self._rim_circle: Optional[Tuple[int, int, int]] = None
        self._rim_hue: float = -1.0
        self._prev_gray: Optional[np.ndarray] = None
        self._identity_bootstrap: Dict[int, List[float]] = {}
        self._rail_mask: Optional[np.ndarray] = None
        self._polygon_points: Optional[List[Tuple[int, int]]] = None
        self._rail_hue_learned: bool = False
        self._mm_per_pixel: float = 0.0
        self._roi_miss_count: Dict[int, int] = {}

    # ------------------------------------------------------------------ #
    #  Arena ROI                                                          #
    # ------------------------------------------------------------------ #

    def _recompute_mm_per_pixel(self) -> None:
        """Derive the pixel-to-mm scale from the best available boundary.

        Priority: rim_circle (Hough-detected rim) > polygon bounding circle
        > arena_roi_low (outer ROI) > arena_roi (fallback).
        """
        diameter_px = 0.0

        if self._rim_circle is not None:
            diameter_px = self._rim_circle[2] * 2.0
        elif self._polygon_points is not None and len(self._polygon_points) >= 3:
            pts = np.array(self._polygon_points, dtype=np.float32)
            _, radius = cv2.minEnclosingCircle(pts)
            diameter_px = radius * 2.0
        elif self._arena_roi_low is not None:
            diameter_px = self._arena_roi_low[2] * 2.0
        elif self._arena_roi is not None:
            diameter_px = self._arena_roi[2] * 2.0

        if diameter_px > 0:
            diameter_mm = float(getattr(config, "STADIUM_DIAMETER_MM", 400.0))
            self._mm_per_pixel = diameter_mm / diameter_px

    def set_arena_roi(self, cx: int, cy: int, r: int) -> None:
        ox = int(getattr(config, "ARENA_ROI_OFFSET_X", 0))
        oy = int(getattr(config, "ARENA_ROI_OFFSET_Y", 0))
        self._arena_roi = (int(cx) + ox, int(cy) + oy, int(r))
        self._arena_roi_high = None
        self._arena_roi_low = None
        self._recompute_mm_per_pixel()

    def set_arena_roi_dual(
        self,
        cx: int,
        cy: int,
        r_high: int,
        r_low: int,
    ) -> None:
        ox = int(getattr(config, "ARENA_ROI_OFFSET_X", 0))
        oy = int(getattr(config, "ARENA_ROI_OFFSET_Y", 0))
        cxi, cyi = int(cx) + ox, int(cy) + oy
        self._arena_roi = (cxi, cyi, r_low)
        self._arena_roi_high = (cxi, cyi, r_high)
        self._arena_roi_low = (cxi, cyi, r_low)
        self._recompute_mm_per_pixel()

    def set_arena_roi_high_only(self, cx: int, cy: int, r: int) -> None:
        """Red circle only; no green. Use polygon for full area when set."""
        ox = int(getattr(config, "ARENA_ROI_OFFSET_X", 0))
        oy = int(getattr(config, "ARENA_ROI_OFFSET_Y", 0))
        cxi, cyi = int(cx) + ox, int(cy) + oy
        self._arena_roi = (cxi, cyi, r)
        self._arena_roi_high = (cxi, cyi, r)
        self._arena_roi_low = None
        self._recompute_mm_per_pixel()

    def get_arena_roi(self) -> Optional[Tuple[int, int, int]]:
        return self._arena_roi

    def get_arena_roi_high(self) -> Optional[Tuple[int, int, int]]:
        return self._arena_roi_high

    def get_arena_roi_low(self) -> Optional[Tuple[int, int, int]]:
        return self._arena_roi_low

    def get_rim_circle(self) -> Optional[Tuple[int, int, int]]:
        return self._rim_circle

    @property
    def mm_per_pixel(self) -> float:
        return self._mm_per_pixel

    def get_arena_center_px(self) -> Optional[Tuple[int, int]]:
        """Return the stadium center in pixels using the best available boundary."""
        if self._rim_circle is not None:
            return (self._rim_circle[0], self._rim_circle[1])
        if self._polygon_points is not None and len(self._polygon_points) >= 3:
            pts = np.array(self._polygon_points, dtype=np.float32)
            (cx, cy), _ = cv2.minEnclosingCircle(pts)
            return (int(cx), int(cy))
        roi = self._arena_roi_low or self._arena_roi
        if roi is not None:
            return (roi[0], roi[1])
        return None

    def get_arena_radius_px(self) -> float:
        """Return the stadium radius in pixels using the best available boundary."""
        if self._rim_circle is not None:
            return float(self._rim_circle[2])
        if self._polygon_points is not None and len(self._polygon_points) >= 3:
            pts = np.array(self._polygon_points, dtype=np.float32)
            _, radius = cv2.minEnclosingCircle(pts)
            return float(radius)
        roi = self._arena_roi_low or self._arena_roi
        if roi is not None:
            return float(roi[2])
        return 0.0

    def _inside_roi(self, x: float, y: float) -> bool:
        if self._polygon_points is not None:
            pts = np.array(self._polygon_points, dtype=np.int32)
            return cv2.pointPolygonTest(pts, (float(x), float(y)), False) >= 0
        roi = self._arena_roi_low if self._arena_roi_low else self._arena_roi
        if roi is None:
            return True
        rcx, rcy, rr = roi
        return distance((x, y), (rcx, rcy)) <= rr

    def _inside_roi_high(self, x: float, y: float) -> bool:
        if self._arena_roi_high is None:
            return False
        rcx, rcy, rr = self._arena_roi_high
        return distance((x, y), (rcx, rcy)) <= rr

    def _near_polygon_edge(self, x: float, y: float) -> bool:
        """True if (x,y) is inside polygon but within margin px of the edge (rail reflections)."""
        if self._polygon_points is None:
            return False
        margin = float(getattr(config, "POLYGON_EDGE_MARGIN", 0))
        if margin <= 0:
            return False
        pts = np.array(self._polygon_points, dtype=np.int32)
        dist = cv2.pointPolygonTest(pts, (float(x), float(y)), True)
        return 0 < dist < margin

    def _near_rim(self, x: float, y: float) -> bool:
        """True if (x,y) is in the outer band where the green rail sits."""
        if self._polygon_points is not None:
            return False
        frac = float(getattr(config, "REJECT_NEAR_RIM_FRACTION", 0))
        if frac <= 0:
            return False
        stadium = self._rim_circle or self._arena_roi_low or self._arena_roi
        if stadium is None:
            return False
        rcx, rcy, rr = stadium
        d = distance((x, y), (rcx, rcy))
        return d >= rr * (1.0 - frac)

    def build_rail_mask(self, frame: np.ndarray) -> bool:
        """Build mask of green rail region from frame. Stadium is static. Returns True if built."""
        if not getattr(config, "RAIL_MASK_ENABLED", False):
            return False
        stadium = self._rim_circle or self._arena_roi_low or self._arena_roi
        if stadium is None:
            return False
        rcx, rcy, rr = stadium
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        H, S = hsv[:, :, 0], hsv[:, :, 1]
        h_lo = int(getattr(config, "RAIL_MASK_HUE_LO", 35))
        h_hi = int(getattr(config, "RAIL_MASK_HUE_HI", 95))
        sat_min = int(getattr(config, "RAIL_MASK_SAT_MIN", 55))
        inner_frac = float(getattr(config, "RAIL_MASK_ANNULUS_INNER", 0.75))
        outer_scale = float(getattr(config, "RAIL_MASK_OUTER_SCALE", 1.05))
        inner_r = int(rr * inner_frac)
        outer_r = int(rr * outer_scale)
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((X - rcx) ** 2 + (Y - rcy) ** 2)
        annulus = (dist >= inner_r) & (dist <= outer_r)
        if h_lo <= h_hi:
            h_mask = (H >= h_lo) & (H <= h_hi)
        else:
            h_mask = (H >= h_lo) | (H <= h_hi)
        rail = (annulus & h_mask & (S >= sat_min)).astype(np.uint8) * 255
        kernel = np.ones((5, 5), np.uint8)
        rail = cv2.morphologyEx(rail, cv2.MORPH_CLOSE, kernel)
        rail = cv2.morphologyEx(rail, cv2.MORPH_OPEN, kernel)
        if np.sum(rail > 0) < 100:
            return False
        self._rail_mask = rail
        save_path = getattr(config, "RAIL_MASK_SAVE_PATH", "")
        if save_path:
            parent = os.path.dirname(save_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            if cv2.imwrite(save_path, rail):
                print(f"Rail mask saved to {save_path}")
        return True

    def get_rail_mask(self) -> Optional[np.ndarray]:
        """Return the rail mask if built (for debug overlay)."""
        return self._rail_mask

    def set_rail_mask_from_polygon(
        self,
        frame_shape: Tuple[int, ...],
        points: List[Tuple[int, int]],
        *,
        save_path: str = "",
    ) -> bool:
        """
        Set rail mask and tracking ROI from polygon points.
        Polygon = tracking area (inside = where beys are). Rail mask = outside polygon (zero S there).
        """
        if len(points) < 3:
            return False
        h, w = frame_shape[:2]
        self._polygon_points = list(points)
        play_area = np.zeros((h, w), dtype=np.uint8)
        pts = np.array(points, dtype=np.int32)
        cv2.fillPoly(play_area, [pts], 255)
        self._rail_mask = np.where(play_area > 0, 0, 255).astype(np.uint8)
        self._recompute_mm_per_pixel()
        if save_path:
            parent = os.path.dirname(save_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            if cv2.imwrite(save_path, self._rail_mask):
                print(f"Rail mask saved to {save_path}")
        return True

    def get_polygon_points(self) -> Optional[List[Tuple[int, int]]]:
        """Return polygon points if set (for debug overlay)."""
        return self._polygon_points

    def auto_detect_roi_from_reference(
        self, reference: np.ndarray, frame: np.ndarray
    ) -> bool:
        ref_hsv = cv2.cvtColor(reference, cv2.COLOR_BGR2HSV)
        sat_mask = ref_hsv[:, :, 1] >= 60
        if np.sum(sat_mask) < 100:
            return False
        rim_hue = float(np.median(ref_hsv[:, :, 0][sat_mask]))

        if config.DEBUG_PRINT:
            print(f"Reference rim hue: {rim_hue:.0f}")

        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        H = frame_hsv[:, :, 0]
        S = frame_hsv[:, :, 1]
        V = frame_hsv[:, :, 2]

        tol = 25
        h_lo, h_hi = rim_hue - tol, rim_hue + tol
        if h_lo < 0:
            h_mask = (H <= h_hi) | (H >= int(180 + h_lo))
        elif h_hi > 180:
            h_mask = (H >= h_lo) | (H <= int(h_hi - 180))
        else:
            h_mask = (H >= h_lo) & (H <= h_hi)

        mask = (h_mask & (S >= 45) & (V >= 40)).astype(np.uint8) * 255

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return False

        areas = [(cv2.contourArea(c), c) for c in contours]
        areas.sort(key=lambda x: x[0], reverse=True)

        if not areas or areas[0][0] < 500:
            return False
        area_thresh = areas[0][0] * 0.1
        valid = [c for a, c in areas if a >= max(area_thresh, 500)]
        if not valid:
            return False

        combined = np.vstack(valid)
        if len(combined) < 10:
            return False

        pts = combined.reshape(-1, 2).astype(np.float64)
        cx, cy, r = _fit_circle_lsq(pts)
        if r <= 0:
            return False

        h, w = frame.shape[:2]
        min_dim = min(h, w)

        if not (0.05 * w < cx < 0.95 * w and 0.05 * h < cy < 0.95 * h):
            return False
        if r < min_dim * 0.1 or r > min_dim * 0.85:
            return False

        angles = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
        quadrant_bits = set()
        for a in angles:
            quadrant_bits.add(int(a // (np.pi / 2)))
        if len(quadrant_bits) < 3:
            return False

        ann_check = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(ann_check, (int(cx), int(cy)), int(r * 1.05), 255, -1)
        cv2.circle(ann_check, (int(cx), int(cy)), int(r * 0.85), 0, -1)
        ann_total = np.sum(ann_check > 0)
        ann_match = np.sum((ann_check > 0) & (mask > 0))
        rim_fill = float(ann_match) / max(ann_total, 1)
        if rim_fill < 0.20:
            return False

        check_r = int(r * 0.6)
        interior_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(interior_mask, (int(cx), int(cy)), check_r, 255, -1)
        interior_v = V[interior_mask > 0]
        interior_s = S[interior_mask > 0]
        if len(interior_v) == 0:
            return False
        mean_v = float(np.mean(interior_v))
        mean_s = float(np.mean(interior_s))
        if mean_v < 120 or mean_s > 80:
            return False

        shrink = float(getattr(config, "ARENA_RIM_SHRINK", 0.70))
        self._rim_circle = (int(cx), int(cy), int(r))
        self._recompute_mm_per_pixel()

        rim_inner = int(r * shrink)
        rim_outer = int(r)
        annulus = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(annulus, (int(cx), int(cy)), rim_outer, 255, -1)
        cv2.circle(annulus, (int(cx), int(cy)), rim_inner, 0, -1)
        ann_sat = S[annulus > 0]
        ann_hue = H[annulus > 0]
        sat_ok = ann_sat >= 45
        if np.sum(sat_ok) > 20:
            self._rim_hue = float(np.median(ann_hue[sat_ok]))
        else:
            self._rim_hue = rim_hue

        r *= shrink
        self.set_arena_roi(int(cx), int(cy), int(r))
        if config.DEBUG_PRINT:
            print(
                f"Auto ROI from reference: center=({int(cx)},{int(cy)}) "
                f"r={int(r)} (rim_hue={self._rim_hue:.0f})"
            )
        return True

    def auto_detect_roi_hough(self, frame: np.ndarray) -> bool:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        h, w = frame.shape[:2]
        min_dim = min(h, w)

        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1.5,
            minDist=min_dim // 2,
            param1=80, param2=60,
            minRadius=int(min_dim * 0.15),
            maxRadius=int(min_dim * 0.85),
        )
        if circles is None:
            return False

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        S, V = hsv[:, :, 1], hsv[:, :, 2]

        best = None
        best_score = -1.0
        for c in circles[0]:
            cx, cy, r = float(c[0]), float(c[1]), float(c[2])
            if not (0.1 * w < cx < 0.9 * w and 0.1 * h < cy < 0.9 * h):
                continue
            check_r = int(r * 0.5)
            imask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(imask, (int(cx), int(cy)), check_r, 255, -1)
            iv = V[imask > 0]
            is_ = S[imask > 0]
            if len(iv) == 0:
                continue
            mv = float(np.mean(iv))
            ms = float(np.mean(is_))
            if mv < 120 or ms > 80:
                continue
            score = r * (mv / 255.0) * (1.0 - ms / 255.0)
            if score > best_score:
                best_score = score
                best = (cx, cy, r)

        if best is None:
            return False

        cx, cy, r = best
        self._rim_circle = (int(cx), int(cy), int(r))
        self._recompute_mm_per_pixel()

        # Learn rim hue from the annulus
        H = hsv[:, :, 0]
        rim_inner = int(r * 0.80)
        rim_outer = int(r)
        annulus = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(annulus, (int(cx), int(cy)), rim_outer, 255, -1)
        cv2.circle(annulus, (int(cx), int(cy)), rim_inner, 0, -1)
        ann_sat = S[annulus > 0]
        ann_hue = H[annulus > 0]
        sat_ok = ann_sat >= 45
        if np.sum(sat_ok) > 20:
            self._rim_hue = float(np.median(ann_hue[sat_ok]))

        shrink = float(getattr(config, "ARENA_RIM_SHRINK", 0.70))
        r *= shrink
        self.set_arena_roi(int(cx), int(cy), int(r))
        if config.DEBUG_PRINT:
            print(
                f"Auto ROI from Hough: center=({int(cx)},{int(cy)}) "
                f"r={int(r)} rim_hue={self._rim_hue:.0f}"
            )
        return True

    # ------------------------------------------------------------------ #
    #  Hough circle detection                                              #
    # ------------------------------------------------------------------ #

    def _detect_circles_hough(
        self,
        channel: np.ndarray,
        frame_shape: Tuple[int, ...],
        *,
        offset: Tuple[int, int] = (0, 0),
    ) -> List[Tuple[Tuple[int, int], int]]:
        """Run HoughCircles on a single-channel image (saturation or grayscale).

        Returns list of ((cx, cy), radius) inside the ROI.
        When *offset* is non-zero, adds it to every coordinate before the
        ROI check (used for detection on cropped sub-images).
        """
        h, w = frame_shape[:2]
        min_dim = min(h, w)
        min_radius = max(config.HOUGH_MIN_RADIUS, int(0.02 * min_dim))
        max_radius = max(
            min_radius + 1,
            min(config.HOUGH_MAX_RADIUS, int(0.18 * min_dim)),
        )
        blurred = cv2.GaussianBlur(
            channel, config.GAUSSIAN_BLUR_KSIZE, config.GAUSSIAN_BLUR_SIGMA
        )
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            config.HOUGH_DP,
            config.HOUGH_MIN_DIST,
            param1=config.HOUGH_PARAM1,
            param2=config.HOUGH_PARAM2,
            minRadius=min_radius,
            maxRadius=max_radius,
        )
        if circles is None:
            return []
        ox, oy = offset
        out = [((int(c[0]) + ox, int(c[1]) + oy), int(c[2])) for c in circles[0]]
        return [c for c in out if self._inside_roi(c[0][0], c[0][1])]

    def _detect_circles_hough_gray(
        self,
        gray: np.ndarray,
        frame_shape: Tuple[int, ...],
    ) -> List[Tuple[Tuple[int, int], int]]:
        """HoughCircles on grayscale with relaxed params for low-contrast beys.

        Uses lighter blur and a lower Canny threshold than the saturation
        path so that subtle edges (metallic/transparent/white beys on a
        white floor) are not suppressed.
        """
        h, w = frame_shape[:2]
        min_dim = min(h, w)
        min_radius = max(config.HOUGH_MIN_RADIUS, int(0.02 * min_dim))
        max_radius = max(
            min_radius + 1,
            min(config.HOUGH_MAX_RADIUS, int(0.18 * min_dim)),
        )
        blur_k = tuple(getattr(config, "HOUGH_GRAY_BLUR_KSIZE", (7, 7)))
        blur_s = float(getattr(config, "HOUGH_GRAY_BLUR_SIGMA", 1.5))
        param1 = int(getattr(config, "HOUGH_GRAY_PARAM1", 50))
        blurred = cv2.GaussianBlur(gray, blur_k, blur_s)
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            config.HOUGH_DP,
            config.HOUGH_MIN_DIST,
            param1=param1,
            param2=config.HOUGH_PARAM2,
            minRadius=min_radius,
            maxRadius=max_radius,
        )
        if circles is None:
            return []
        out = [((int(c[0]), int(c[1])), int(c[2])) for c in circles[0]]
        return [c for c in out if self._inside_roi(c[0][0], c[0][1])]

    # ------------------------------------------------------------------ #
    #  Contour-based detection                                              #
    # ------------------------------------------------------------------ #

    def _detect_contours(
        self,
        channel: np.ndarray,
        frame_shape: Tuple[int, ...],
        *,
        offset: Tuple[int, int] = (0, 0),
    ) -> List[Tuple[Tuple[int, int], int]]:
        """Find bey candidates via thresholding + contour analysis.

        Takes the same saturation channel as _detect_circles_hough and
        returns the same format: list of ((cx, cy), radius).
        When *offset* is non-zero, adds it to every coordinate before the
        ROI check (used for detection on cropped sub-images).
        """
        threshold = int(getattr(config, "CONTOUR_SAT_THRESHOLD", 35))
        min_area = int(getattr(config, "CONTOUR_MIN_AREA", 150))
        max_area = int(getattr(config, "CONTOUR_MAX_AREA", 8000))
        min_circ = float(getattr(config, "CONTOUR_MIN_CIRCULARITY", 0.25))
        morph_k = int(getattr(config, "CONTOUR_MORPH_KSIZE", 5))

        _, binary = cv2.threshold(channel, threshold, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_k, morph_k))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        ox, oy = offset
        out: List[Tuple[Tuple[int, int], int]] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue

            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = (4.0 * np.pi * area) / (perimeter * perimeter)
            if circularity < min_circ:
                continue

            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"]) + ox
            cy = int(M["m01"] / M["m00"]) + oy

            radius = int(np.sqrt(area / np.pi))

            if self._inside_roi(cx, cy):
                out.append(((cx, cy), radius))

        return out

    # ------------------------------------------------------------------ #
    #  ROI-based detection                                                  #
    # ------------------------------------------------------------------ #

    def _detect_in_roi(
        self,
        channel: np.ndarray,
        center: Tuple[float, float],
        roi_size: int,
    ) -> List[Tuple[Tuple[int, int], int]]:
        """Run detection on a small square crop around *center*.

        Returns circles in **global** frame coordinates.
        """
        fh, fw = channel.shape[:2]
        half = roi_size // 2
        ci, cj = int(center[0]), int(center[1])

        x1 = max(0, ci - half)
        y1 = max(0, cj - half)
        x2 = min(fw, ci + half)
        y2 = min(fh, cj + half)

        crop = channel[y1:y2, x1:x2]
        if crop.size == 0:
            return []

        offset = (x1, y1)
        method = getattr(config, "DETECTION_METHOD", "hough")
        if method == "contour":
            return self._detect_contours(crop, crop.shape, offset=offset)
        return self._detect_circles_hough(crop, crop.shape, offset=offset)

    def _detect_candidates(
        self,
        channel: np.ndarray,
        frame_shape: Tuple[int, ...],
        *,
        gray: Optional[np.ndarray] = None,
    ) -> List[Tuple[Tuple[int, int], int]]:
        """Two-tier detection: fast ROI per tracked bey, full-frame fallback.

        Tier 1 (fast): for each tracked bey, crop a small window around its
        linearly-extrapolated position and detect only there.
        Tier 2 (fallback): runs on the full frame when a ROI search misses
        or when new bey discovery is needed.

        Returns the combined, deduplicated list of circles.
        """
        roi_enabled = getattr(config, "ROI_ENABLED", False)
        roi_size = int(getattr(config, "ROI_SIZE", 0))
        fallback_limit = int(getattr(config, "ROI_FALLBACK_FRAMES", 3))

        if not roi_enabled or roi_size <= 0 or not self._bey:
            return self._detect_full_frame(channel, frame_shape, gray=gray)

        nominal_dt = 1.0 / max(getattr(config, "TARGET_FPS", 60), 1)
        need_full_frame = len(self._bey) < config.MAX_BEY_COUNT

        roi_circles: List[Tuple[Tuple[int, int], int]] = []
        for b in self._bey:
            pred_x = b.position[0] + b.velocity[0] * nominal_dt
            pred_y = b.position[1] + b.velocity[1] * nominal_dt
            found = self._detect_in_roi(channel, (pred_x, pred_y), roi_size)
            if found:
                roi_circles.extend(found)
                self._roi_miss_count[b.id] = 0
            else:
                misses = self._roi_miss_count.get(b.id, 0) + 1
                self._roi_miss_count[b.id] = misses
                if misses >= fallback_limit:
                    need_full_frame = True

        if not need_full_frame:
            return roi_circles

        full_circles = self._detect_full_frame(channel, frame_shape, gray=gray)
        if not roi_circles:
            return full_circles

        combined = list(roi_circles)
        dedup_dist = float(getattr(config, "CANDIDATE_MIN_SEPARATION", 28))
        for c in full_circles:
            if not any(
                distance(c[0], ec[0]) < dedup_dist for ec in combined
            ):
                combined.append(c)
        return combined

    def _detect_full_frame(
        self,
        channel: np.ndarray,
        frame_shape: Tuple[int, ...],
        *,
        gray: Optional[np.ndarray] = None,
    ) -> List[Tuple[Tuple[int, int], int]]:
        """Run detection on the entire channel.

        When using contour detection and fewer circles than MAX_BEY_COUNT are
        found, falls back to HoughCircles on grayscale to catch
        low-saturation beys that are invisible in the saturation channel.
        """
        method = getattr(config, "DETECTION_METHOD", "hough")
        if method == "contour":
            circles = self._detect_contours(channel, frame_shape)
            if gray is not None and len(circles) < config.MAX_BEY_COUNT:
                hough_extra = self._detect_circles_hough_gray(gray, frame_shape)
                dedup_dist = float(
                    getattr(config, "CANDIDATE_MIN_SEPARATION", 28)
                )
                for c in hough_extra:
                    if not any(
                        distance(c[0], ec[0]) < dedup_dist for ec in circles
                    ):
                        circles.append(c)
            return circles
        return self._detect_circles_hough(channel, frame_shape)

    # ------------------------------------------------------------------ #
    #  Color sampling                                                      #
    # ------------------------------------------------------------------ #

    def _sample_center_color(
        self, frame: np.ndarray, cx: int, cy: int, r: int
    ) -> float:
        """Median hue of saturated pixels in the center chip. Returns -1 on failure."""
        fh, fw = frame.shape[:2]
        sample_r = max(5, int(r * config.COLOR_SAMPLE_RADIUS_FRAC))

        y1, y2 = max(0, cy - sample_r), min(fh, cy + sample_r + 1)
        x1, x2 = max(0, cx - sample_r), min(fw, cx + sample_r + 1)
        patch = frame[y1:y2, x1:x2]
        if patch.size == 0:
            return -1.0

        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        ph, pw = patch.shape[:2]
        cy_l, cx_l = cy - y1, cx - x1
        Y, X = np.ogrid[:ph, :pw]
        cmask = ((X - cx_l) ** 2 + (Y - cy_l) ** 2) <= sample_r ** 2

        hues = hsv[:, :, 0][cmask]
        sats = hsv[:, :, 1][cmask]
        vals = hsv[:, :, 2][cmask]
        if len(hues) == 0:
            return -1.0

        val_min = int(getattr(config, "COLOR_VAL_MIN", 0))
        colored = sats >= config.COLOR_SAT_MIN
        if val_min > 0:
            colored = colored & (vals >= val_min)
        n_colored = int(np.sum(colored))
        min_fill = getattr(config, "COLOR_CENTER_MIN_FILL", 0.15)
        strict_ok = n_colored >= 5 and n_colored / len(sats) >= min_fill
        if strict_ok:
            return float(np.median(hues[colored]))

        # Relaxed fallback: halve sat minimum to catch barely-colored beys
        relaxed_sat = max(5, config.COLOR_SAT_MIN // 2)
        relaxed = sats >= relaxed_sat
        if val_min > 0:
            relaxed = relaxed & (vals >= val_min)
        n_relaxed = int(np.sum(relaxed))
        if n_relaxed >= 5 and n_relaxed / len(sats) >= min_fill:
            return float(np.median(hues[relaxed]))

        # Last resort for colorless/metallic/transparent beys:
        # use median hue of all bright-enough pixels regardless of saturation.
        # Only accept when most of the center patch is bright (not dark noise).
        bright_thresh = max(val_min, 30)
        bright = vals >= bright_thresh
        n_bright = int(np.sum(bright))
        if n_bright >= 5 and n_bright / len(vals) >= 0.3:
            return float(np.median(hues[bright]))
        return -1.0

    # ------------------------------------------------------------------ #
    #  Motion check via frame differencing                                 #
    # ------------------------------------------------------------------ #

    def _has_motion_at(
        self, gray: np.ndarray, cx: int, cy: int, radius: float
    ) -> bool:
        """True if there is local pixel change between current and previous frame."""
        if self._prev_gray is None:
            return False
        mult = getattr(config, "MOTION_PATCH_RADIUS_MULT", 1.0)
        r = max(int(radius * mult), 6)
        h, w = gray.shape
        y1, y2 = max(0, cy - r), min(h, cy + r + 1)
        x1, x2 = max(0, cx - r), min(w, cx + r + 1)
        if y2 <= y1 or x2 <= x1:
            return False
        diff = cv2.absdiff(gray[y1:y2, x1:x2], self._prev_gray[y1:y2, x1:x2])
        threshold = getattr(config, "MOTION_THRESHOLD", 12)
        return float(np.mean(diff)) > threshold

    # ------------------------------------------------------------------ #
    #  Candidate matching                                                  #
    # ------------------------------------------------------------------ #

    def _get_identity_hue(self, b: BeyState) -> float:
        """Hue used for identity matching; color_hue_origin once bootstrap done."""
        return b.color_hue_origin if b.color_hue_origin >= 0 else b.color_hue

    def _match_candidates(
        self,
        candidates: List[Tuple[float, float, float, float]],
    ) -> Tuple[Dict[int, Tuple[float, float, float, float]], List[int]]:
        """Match Hough candidates to tracked beys by position + identity hue.

        Uses Kalman prediction distance and identity hue to avoid swapping bey1/bey2
        when they cross. Identity hue is sampled from center chip at first capture.
        """
        if not self._bey or not candidates:
            return {}, list(range(len(candidates)))

        base_dist = getattr(config, "MATCH_MAX_DISTANCE", 200)
        fast_dist = getattr(config, "MATCH_MAX_DISTANCE_FAST", 0)
        fast_thresh = float(getattr(config, "MATCH_FAST_SPEED_THRESHOLD", 40))
        identity_w = getattr(config, "MATCH_IDENTITY_WEIGHT", 4.0)
        max_drift = getattr(config, "IDENTITY_HUE_MAX_DRIFT", 0)

        pairs: List[Tuple[float, int, int]] = []
        for bi, b in enumerate(self._bey):
            pred = b.kalman_predict()
            max_dist = base_dist
            if fast_dist > 0 and b.speed >= fast_thresh:
                max_dist = fast_dist
            identity_hue = self._get_identity_hue(b)
            for ci, (cx, cy, _r, hue) in enumerate(candidates):
                d = distance((cx, cy), pred)
                if d > max_dist:
                    continue
                hd = _hue_distance(hue, identity_hue) if identity_hue >= 0 else 0
                if max_drift > 0 and identity_hue >= 0 and hd > max_drift:
                    continue
                score = d + hd * identity_w
                if getattr(config, "PREFER_HIGH_PRIORITY", False) and self._arena_roi_high:
                    if self._inside_roi_high(cx, cy):
                        score -= 150
                pairs.append((score, bi, ci))

        pairs.sort()

        matched: Dict[int, Tuple[float, float, float, float]] = {}
        used_beys: set = set()
        used_cands: set = set()

        for _score, bi, ci in pairs:
            if bi in used_beys or ci in used_cands:
                continue
            bey = self._bey[bi]
            matched[bey.id] = candidates[ci]
            used_beys.add(bi)
            used_cands.add(ci)

        unmatched = [i for i in range(len(candidates)) if i not in used_cands]
        return matched, unmatched

    # ------------------------------------------------------------------ #
    #  Color model update                                                  #
    # ------------------------------------------------------------------ #

    def _blend_color(self, bey: BeyState, observed_hue: float) -> None:
        """Slowly blend the tracked hue towards the observed hue."""
        if _hue_distance(observed_hue, bey.color_hue) > config.COLOR_HUE_TOLERANCE:
            return
        alpha = config.COLOR_ADAPT_RATE
        r_old = bey.color_hue * np.pi / 90.0
        r_new = observed_hue * np.pi / 90.0
        s = (1 - alpha) * np.sin(r_old) + alpha * np.sin(r_new)
        c = (1 - alpha) * np.cos(r_old) + alpha * np.cos(r_new)
        candidate = float((np.arctan2(s, c) % (2 * np.pi)) * 90.0 / np.pi)
        if bey.color_hue_origin >= 0:
            if _hue_distance(candidate, bey.color_hue_origin) > config.COLOR_HUE_TOLERANCE:
                return
        bey.color_hue = candidate

    def _build_dome_mask(
        self, hsv: np.ndarray, s_channel: np.ndarray
    ) -> Optional[np.ndarray]:
        """Build dome exclusion mask (glare + wedge). Returns uint8 255=excluded, or None."""
        h, w = hsv.shape[:2]
        out = np.zeros((h, w), dtype=np.uint8)
        any_mask = False
        if getattr(config, "DOME_GLARE_ENABLED", False):
            v = hsv[:, :, 2]
            s = s_channel
            v_min = int(getattr(config, "DOME_GLARE_V_MIN", 220))
            s_max = int(getattr(config, "DOME_GLARE_S_MAX", 40))
            glare = ((v >= v_min) & (s < s_max)).astype(np.uint8) * 255
            out = np.maximum(out, glare)
            any_mask = any_mask or np.any(glare > 0)
        if getattr(config, "DOME_EXCLUDE_WEDGE_ENABLED", False):
            start_deg = float(getattr(config, "DOME_EXCLUDE_ANGLE_START", -1))
            end_deg = float(getattr(config, "DOME_EXCLUDE_ANGLE_END", -1))
            if start_deg >= 0 and end_deg >= 0:
                center = self._rim_circle or self._arena_roi_high or self._arena_roi
                if center is not None:
                    cx_roi, cy_roi, _ = center
                    Y, X = np.ogrid[:h, :w]
                    dy = Y.astype(np.float32) - cy_roi
                    dx = X.astype(np.float32) - cx_roi
                    angle_deg = (np.degrees(np.arctan2(dx, -dy)) + 360) % 360
                    if start_deg <= end_deg:
                        wedge = (angle_deg >= start_deg) & (angle_deg <= end_deg)
                    else:
                        wedge = (angle_deg >= start_deg) | (angle_deg <= end_deg)
                    wedge_u8 = wedge.astype(np.uint8) * 255
                    out = np.maximum(out, wedge_u8)
                    any_mask = True
        if not any_mask and (getattr(config, "DOME_GLARE_ENABLED", False) or getattr(config, "DOME_EXCLUDE_WEDGE_ENABLED", False)):
            return out
        return out if any_mask else None

    def get_dome_mask(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Build dome exclusion mask for snapshot/preview. Returns uint8 255=excluded."""
        if getattr(config, "HOUGH_DETECTION_CHANNEL", "grayscale") != "saturation":
            return None
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        s = hsv[:, :, 1].astype(np.float32)
        s *= float(getattr(config, "HOUGH_SAT_SCALE", 1.0))
        s = np.clip(s, 0, 255).astype(np.uint8)
        floor = int(getattr(config, "HOUGH_SAT_FLOOR", 0))
        if floor > 0:
            s = np.where(s >= floor, s, 0).astype(np.uint8)
        if getattr(config, "HOUGH_SAT_CLAHE_ENABLED", False):
            clahe = cv2.createCLAHE(
                clipLimit=float(getattr(config, "HOUGH_SAT_CLAHE_CLIP", 2.5)),
                tileGridSize=getattr(config, "HSV_CLAHE_TILE", (8, 8)),
            )
            s = clahe.apply(s)
        return self._build_dome_mask(hsv, s)

    # ------------------------------------------------------------------ #
    #  Main update                                                         #
    # ------------------------------------------------------------------ #

    def update(self, frame: np.ndarray, dt: float) -> None:
        """One tracking cycle: Hough -> color -> match -> Kalman."""
        self._frame_index += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if getattr(config, "HOUGH_DETECTION_CHANNEL", "grayscale") == "saturation":
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            s = hsv[:, :, 1].astype(np.float32)
            s *= float(getattr(config, "HOUGH_SAT_SCALE", 1.0))
            s = np.clip(s, 0, 255).astype(np.uint8)
            floor = int(getattr(config, "HOUGH_SAT_FLOOR", 0))
            if floor > 0:
                s = np.where(s >= floor, s, 0).astype(np.uint8)
            use_clahe = (
                getattr(config, "HOUGH_SAT_CLAHE_ENABLED", False)
                and getattr(config, "DETECTION_METHOD", "hough") != "contour"
            )
            if use_clahe:
                clahe = cv2.createCLAHE(
                    clipLimit=float(getattr(config, "HOUGH_SAT_CLAHE_CLIP", 2.5)),
                    tileGridSize=getattr(config, "HSV_CLAHE_TILE", (8, 8)),
                )
                s = clahe.apply(s)
            if self._rail_mask is not None:
                s[self._rail_mask > 0] = 0
            dome_mask = self._build_dome_mask(hsv, s)
            if dome_mask is not None:
                s[dome_mask > 0] = 0
            channel = s
        else:
            channel = gray

        if (
            self._rail_mask is not None
            and self._polygon_points is not None
            and not self._rail_hue_learned
            and self._rim_hue < 0
        ):
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            H = hsv[:, :, 0][self._rail_mask > 0]
            S = hsv[:, :, 1][self._rail_mask > 0]
            sat_ok = S >= 50
            if np.sum(sat_ok) > 30:
                self._rim_hue = float(np.median(H[sat_ok]))
                self._rail_hue_learned = True
                if config.DEBUG_PRINT:
                    print(f"Learned rail hue from polygon: {self._rim_hue:.0f}")

        # --- 1. Detect bey candidates (ROI fast-path when tracking) ---
        circles = self._detect_candidates(channel, frame.shape, gray=gray)

        # --- 2. Sample center color, reject rim-coloured circles ---
        all_candidates: List[Tuple[float, float, float, float]] = []
        for (cx, cy), r in circles:
            hue = self._sample_center_color(frame, cx, cy, r)
            if hue < 0:
                continue
            if self._rim_hue >= 0:
                if _hue_distance(hue, self._rim_hue) < config.COLOR_MIN_HUE_SEPARATION:
                    continue
            reject_ranges = getattr(config, "REJECT_HUE_RANGES", [])
            if any(lo <= hue <= hi for lo, hi in reject_ranges):
                continue
            if self._near_rim(cx, cy):
                continue
            if self._near_polygon_edge(cx, cy):
                if getattr(config, "RAIL_TRACKING_ALLOW_EDGE", False) and self._bey:
                    max_dist = getattr(config, "MATCH_MAX_DISTANCE", 200)
                    near_tracked = any(
                        distance((cx, cy), b.position) < max_dist for b in self._bey
                    )
                    if not near_tracked:
                        continue
                else:
                    continue
            all_candidates.append((float(cx), float(cy), float(r), hue))

        # Deduplicate: overlapping circles = same physical bey, keep one
        min_sep = getattr(config, "CANDIDATE_MIN_SEPARATION", 28)
        deduped: List[Tuple[float, float, float, float]] = []
        for c in all_candidates:
            cx, cy, r, hue = c
            overlaps = any(
                distance((cx, cy), (ox, oy)) < max(min_sep, r + or_)
                for ox, oy, or_, _ in deduped
            )
            if not overlaps:
                deduped.append(c)
        all_candidates = deduped

        if self._arena_roi_high is not None:
            high = [c for c in all_candidates if self._inside_roi_high(c[0], c[1])]
            low_only = [
                c for c in all_candidates
                if not self._inside_roi_high(c[0], c[1])
            ]
            candidates = high if len(high) >= 2 else high + low_only
        else:
            candidates = all_candidates
        self._last_circle_count = len(candidates)

        # --- 3. Kalman predict for all tracked beys ---
        for b in self._bey:
            b.kalman_predict()

        # --- 4. Match candidates to tracked beys ---
        matched, unmatched_indices = self._match_candidates(candidates)

        # --- 5. Update tracked beys ---
        for b in self._bey:
            if b.id in matched:
                cx, cy, r, hue = matched[b.id]
                b.update_from_raw((cx, cy), r, dt)
                if config.COLOR_ADAPT_RATE > 0:
                    self._blend_color(b, hue)
                bootstrap_frames = getattr(config, "IDENTITY_BOOTSTRAP_FRAMES", 0)
                if bootstrap_frames > 0 and b.id in self._identity_bootstrap:
                    samples = self._identity_bootstrap[b.id]
                    samples.append(hue)
                    if len(samples) >= bootstrap_frames:
                        b.color_hue_origin = float(np.median(samples))
                        del self._identity_bootstrap[b.id]
                        if config.DEBUG_PRINT:
                            print(
                                f"Bey#{b.id} identity locked: hue={b.color_hue_origin:.0f} "
                                f"(from {len(samples)} samples)"
                            )
            else:
                b.keep_previous_position()

        # --- 6. Register new beys from unmatched circles ---
        # Rim-coloured circles are already filtered out in step 2.
        # A new bey just needs to be a coloured circle that is spatially
        # separated from existing tracked beys.  No colour uniqueness is
        # required -- two beys with identical chip colours are tracked
        # independently by position via Kalman.
        min_sep = getattr(config, "DISCOVERY_MIN_SEPARATION", 40)
        if len(self._bey) < config.MAX_BEY_COUNT:
            for ci in unmatched_indices:
                if len(self._bey) >= config.MAX_BEY_COUNT:
                    break
                cx, cy, r, hue = candidates[ci]

                # Must be far enough from every already-tracked bey
                too_close = False
                for b in self._bey:
                    sep = max(min_sep, r + b.radius)
                    if distance((cx, cy), b.position) < sep:
                        too_close = True
                        break
                if too_close:
                    continue

                new_bey = BeyState(
                    id=self._next_id,
                    position=(cx, cy),
                    previous_position=(cx, cy),
                    velocity=(0.0, 0.0),
                    speed=0.0,
                    radius=r,
                    color_hue=hue,
                    color_hue_origin=hue,
                    radius_origin=r,
                )
                self._bey.append(new_bey)
                if getattr(config, "CIRCULAR_PREDICTION_ENABLED", False):
                    new_bey.circular_history.append((cx, cy))
                bootstrap_frames = getattr(config, "IDENTITY_BOOTSTRAP_FRAMES", 0)
                if bootstrap_frames > 0:
                    self._identity_bootstrap[new_bey.id] = [hue]
                self._next_id += 1
                self._bootstrapped = True

                if config.DEBUG_PRINT:
                    print(
                        f"New bey#{new_bey.id}: hue={hue:.0f} "
                        f"pos=({cx:.0f},{cy:.0f}) r={r:.0f}"
                    )

        # --- 6a. Prefer high-priority: replace edge bey with unmatched center candidate ---
        # Skip when single-bey: if already tracking one, do not switch to another circle.
        if (
            getattr(config, "PREFER_HIGH_PRIORITY", False)
            and self._arena_roi_high is not None
            and len(self._bey) == config.MAX_BEY_COUNT
            and config.MAX_BEY_COUNT > 1
        ):
            hcx, hcy, hr = self._arena_roi_high
            high_unmatched = [
                (ci, candidates[ci])
                for ci in unmatched_indices
                if self._inside_roi_high(candidates[ci][0], candidates[ci][1])
            ]
            if high_unmatched:
                edge_beys = [
                    b for b in self._bey
                    if not self._inside_roi_high(b.position[0], b.position[1])
                ]
            if high_unmatched and edge_beys:
                    replace_bey = max(
                        edge_beys,
                        key=lambda b: (b.frames_since_seen, distance(b.position, (hcx, hcy))),
                    )
                    ci, (cx, cy, r, hue) = high_unmatched[0]
                    too_close = any(
                        distance((cx, cy), b.position) < max(35, r + b.radius)
                        for b in self._bey if b.id != replace_bey.id
                    )
                    if not too_close:
                        self._identity_bootstrap.pop(replace_bey.id, None)
                        self._roi_miss_count.pop(replace_bey.id, None)
                        self._bey.remove(replace_bey)
                        new_bey = BeyState(
                            id=replace_bey.id,
                            position=(cx, cy),
                            previous_position=(cx, cy),
                            velocity=(0.0, 0.0),
                            speed=0.0,
                            radius=r,
                            color_hue=hue,
                            color_hue_origin=hue,
                            radius_origin=r,
                        )
                        self._bey.append(new_bey)
                        if getattr(config, "CIRCULAR_PREDICTION_ENABLED", False):
                            new_bey.circular_history.append((cx, cy))
                        if getattr(config, "IDENTITY_BOOTSTRAP_FRAMES", 0) > 0:
                            self._identity_bootstrap[new_bey.id] = [hue]
                        if config.DEBUG_PRINT:
                            print(
                                f"Replaced bey#{replace_bey.id} with high-priority "
                                f"pos=({cx:.0f},{cy:.0f})"
                            )

        # --- 6b. Clamp velocity to stadium rim (no arrows pointing outside) ---
        stadium = self._rim_circle or self._arena_roi_low or self._arena_roi
        rim_frac = float(getattr(config, "KALMAN_RIM_CLAMP_FRAC", 0.0))
        if stadium is not None and rim_frac > 0:
            for b in self._bey:
                vx, vy = _clamp_velocity_to_rim(
                    b.position, b.velocity, stadium, rim_frac
                )
                if (vx, vy) != b.velocity:
                    b.velocity = (vx, vy)
                    b.speed = float(np.sqrt(vx * vx + vy * vy))
                    if b._kalman is not None:
                        b._kalman.statePost[2] = np.float32(vx)
                        b._kalman.statePost[3] = np.float32(vy)

        # --- 6c. Merge tracked beys that overlap (2 circles on same physical bey) ---
        if len(self._bey) >= 2:
            merge_thresh = getattr(config, "CANDIDATE_MIN_SEPARATION", 28)
            to_drop: set = set()
            for i, b1 in enumerate(self._bey):
                if b1.id in to_drop:
                    continue
                for j, b2 in enumerate(self._bey):
                    if i >= j or b2.id in to_drop:
                        continue
                    d = distance(b1.position, b2.position)
                    if d < max(merge_thresh, b1.radius + b2.radius):
                        in1 = self._arena_roi_high and self._inside_roi_high(b1.position[0], b1.position[1])
                        in2 = self._arena_roi_high and self._inside_roi_high(b2.position[0], b2.position[1])
                        keep = b1 if (b1.frames_since_seen, 0 if in1 else 1) <= (b2.frames_since_seen, 0 if in2 else 1) else b2
                        to_drop.add(b2.id if keep is b1 else b1.id)
            for bid in to_drop:
                self._identity_bootstrap.pop(bid, None)
                self._roi_miss_count.pop(bid, None)
            self._bey = [b for b in self._bey if b.id not in to_drop]

        # --- 6d. Enforce MAX_BEY_COUNT: drop excess if ever over limit ---
        if len(self._bey) > config.MAX_BEY_COUNT:
            self._bey.sort(
                key=lambda b: (
                    b.frames_since_seen,
                    0 if self._arena_roi_high and self._inside_roi_high(b.position[0], b.position[1]) else 1,
                )
            )
            for b in self._bey[config.MAX_BEY_COUNT :]:
                self._identity_bootstrap.pop(b.id, None)
                self._roi_miss_count.pop(b.id, None)
            self._bey = self._bey[: config.MAX_BEY_COUNT]

        # --- 7. Drop beys lost, stationary, or outside polygon ---
        max_drop = getattr(config, "MAX_RECOVERY_FRAMES", 20)
        overlap_dist = float(getattr(config, "OVERLAP_RECOVERY_DISTANCE", 0))
        zero_vel_frames = int(getattr(config, "ZERO_VELOCITY_CLEAR_FRAMES", 0))
        zero_vel_thresh = float(getattr(config, "ZERO_VELOCITY_THRESHOLD", 3.0))
        matched_ids = set(matched.keys())
        dropped_ids = set()
        for b in self._bey:
            if self._polygon_points is not None and not self._inside_roi(b.position[0], b.position[1]):
                dropped_ids.add(b.id)
                if config.DEBUG_PRINT:
                    print(f"Dropped bey#{b.id}: outside polygon (stuck on rail)")
            elif b.frames_since_seen >= max_drop:
                if overlap_dist > 0 and b.id not in matched_ids:
                    near_matched = any(
                        b2.id in matched_ids
                        and distance(b.position, b2.position) < overlap_dist
                        for b2 in self._bey
                        if b2.id != b.id
                    )
                    if near_matched:
                        continue
                dropped_ids.add(b.id)
            elif (
                zero_vel_frames > 0
                and b.frames_stationary >= zero_vel_frames
                and b.speed < zero_vel_thresh
            ):
                dropped_ids.add(b.id)
                if config.DEBUG_PRINT:
                    print(
                        f"Dropped bey#{b.id}: zero velocity for {b.frames_stationary} frames "
                        f"(likely wrong object)"
                    )
        for bid in dropped_ids:
            self._identity_bootstrap.pop(bid, None)
            self._roi_miss_count.pop(bid, None)
        self._bey = [b for b in self._bey if b.id not in dropped_ids]

        if not self._bey:
            self._bootstrapped = False
            return

        # --- 8. Stuck detection ---
        min_speed = getattr(config, "BOOTSTRAP_MIN_SPEED", 5.0)
        max_stuck = getattr(config, "MAX_STUCK_FRAMES", 60)
        if all(b.speed < min_speed for b in self._bey):
            self._frames_all_stuck += 1
        else:
            self._frames_all_stuck = 0

        if self._frames_all_stuck >= max_stuck:
            self._bootstrapped = False
            self._bey.clear()
            self._identity_bootstrap.clear()
            self._roi_miss_count.clear()
            self._frames_all_stuck = 0
            if config.DEBUG_PRINT:
                print(f"All beys stuck for {max_stuck} frames -- clearing")

        # Store for next frame's motion check
        self._prev_gray = gray.copy()

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def get_states(self) -> List[BeyState]:
        return list(self._bey)

    def get_last_circle_count(self) -> int:
        return self._last_circle_count

    def is_bootstrapped(self) -> bool:
        return self._bootstrapped

    def get_debug_masks(self, frame: np.ndarray) -> Dict[int, np.ndarray]:
        """No colour masks in this pipeline -- returns empty dict."""
        return {}

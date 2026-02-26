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

    # ------------------------------------------------------------------ #
    #  Arena ROI                                                          #
    # ------------------------------------------------------------------ #

    def set_arena_roi(self, cx: int, cy: int, r: int) -> None:
        ox = int(getattr(config, "ARENA_ROI_OFFSET_X", 0))
        oy = int(getattr(config, "ARENA_ROI_OFFSET_Y", 0))
        self._arena_roi = (int(cx) + ox, int(cy) + oy, int(r))
        self._arena_roi_high = None
        self._arena_roi_low = None

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

    def get_arena_roi(self) -> Optional[Tuple[int, int, int]]:
        return self._arena_roi

    def get_arena_roi_high(self) -> Optional[Tuple[int, int, int]]:
        return self._arena_roi_high

    def get_arena_roi_low(self) -> Optional[Tuple[int, int, int]]:
        return self._arena_roi_low

    def get_rim_circle(self) -> Optional[Tuple[int, int, int]]:
        return self._rim_circle

    def _inside_roi(self, x: float, y: float) -> bool:
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
        self, gray: np.ndarray, frame_shape: Tuple[int, ...]
    ) -> List[Tuple[Tuple[int, int], int]]:
        """Run HoughCircles on the grayscale image.

        Returns list of ((cx, cy), radius) inside the ROI.
        """
        h, w = frame_shape[:2]
        min_dim = min(h, w)
        min_radius = max(config.HOUGH_MIN_RADIUS, int(0.02 * min_dim))
        max_radius = max(
            min_radius + 1,
            min(config.HOUGH_MAX_RADIUS, int(0.18 * min_dim)),
        )
        blurred = cv2.GaussianBlur(
            gray, config.GAUSSIAN_BLUR_KSIZE, config.GAUSSIAN_BLUR_SIGMA
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
        out = [((int(c[0]), int(c[1])), int(c[2])) for c in circles[0]]
        return [c for c in out if self._inside_roi(c[0][0], c[0][1])]

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
        if len(hues) == 0:
            return -1.0

        colored = sats >= config.COLOR_SAT_MIN
        n_colored = int(np.sum(colored))
        if n_colored < 5:
            return -1.0
        min_fill = getattr(config, "COLOR_CENTER_MIN_FILL", 0.15)
        if n_colored / len(sats) < min_fill:
            return -1.0
        return float(np.median(hues[colored]))

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

        max_dist = getattr(config, "MATCH_MAX_DISTANCE", 200)
        identity_w = getattr(config, "MATCH_IDENTITY_WEIGHT", 4.0)
        max_drift = getattr(config, "IDENTITY_HUE_MAX_DRIFT", 0)

        pairs: List[Tuple[float, int, int]] = []
        for bi, b in enumerate(self._bey):
            pred = b.kalman_predict()
            identity_hue = self._get_identity_hue(b)
            for ci, (cx, cy, _r, hue) in enumerate(candidates):
                d = distance((cx, cy), pred)
                if d > max_dist:
                    continue
                hd = _hue_distance(hue, identity_hue) if identity_hue >= 0 else 0
                if max_drift > 0 and identity_hue >= 0 and hd > max_drift:
                    continue
                score = d + hd * identity_w
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

    # ------------------------------------------------------------------ #
    #  Main update                                                         #
    # ------------------------------------------------------------------ #

    def update(self, frame: np.ndarray, dt: float) -> None:
        """One tracking cycle: Hough -> color -> match -> Kalman."""
        self._frame_index += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # --- 1. Hough circles ---
        circles = self._detect_circles_hough(gray, frame.shape)

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
            all_candidates.append((float(cx), float(cy), float(r), hue))

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
        if (
            getattr(config, "PREFER_HIGH_PRIORITY", False)
            and self._arena_roi_high is not None
            and len(self._bey) == config.MAX_BEY_COUNT
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

        # --- 7. Drop beys lost for too long ---
        max_drop = getattr(config, "MAX_RECOVERY_FRAMES", 20)
        dropped_ids = {b.id for b in self._bey if b.frames_since_seen >= max_drop}
        for bid in dropped_ids:
            self._identity_bootstrap.pop(bid, None)
        self._bey = [b for b in self._bey if b.frames_since_seen < max_drop]

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

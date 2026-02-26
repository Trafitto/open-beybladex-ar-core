"""
Configuration for Beyblade X arena tracking.

Pipeline: Hough circles -> center color sample -> motion check -> Kalman track.

Tuned for: 2 beyblades, very fast motion, white stadium background, steady camera.
"""

# Camera
CAMERA_INDEX = 0          # Webcam device index (0 = default)
TARGET_FPS = 60           # Target frame rate when using live camera

# Hough Circle detection (runs every frame to find bey candidates)
# HOUGH_DP: inverse ratio accumulator/image resolution (1.0-2.0 typical)
# HOUGH_MIN_DIST: minimum px between circle centers; increase to reduce duplicates
# HOUGH_PARAM1: Canny edge high threshold; HOUGH_PARAM2: accumulator threshold
#   lower param2 = more detections but noisier; lower (~14-16) helps when beys blur
# HOUGH_MIN_RADIUS, HOUGH_MAX_RADIUS: px range for bey circle size
HOUGH_DP = 1.2
HOUGH_MIN_DIST = 25
HOUGH_PARAM1 = 80
HOUGH_PARAM2 = 14
HOUGH_MIN_RADIUS = 10
HOUGH_MAX_RADIUS = 20

# Hough input channel: "saturation" (HSV S) or "grayscale"
# Saturation gives better contrast: white floor = low S, colored chips = high S
# Grayscale can blend gray/silver bey parts with white stadium
HOUGH_DETECTION_CHANNEL = "saturation"

# Saturation channel tuning (only when HOUGH_DETECTION_CHANNEL == "saturation")
# HOUGH_SAT_SCALE: multiply S channel (1.0 = no change); >1 boosts colored vs white
# HOUGH_SAT_FLOOR: clip values below this to 0 (suppress near-white noise, 0 = disabled)
# HOUGH_SAT_CLAHE_ENABLED: apply CLAHE to S for uneven lighting across arena
# HOUGH_SAT_CLAHE_CLIP: CLAHE clip limit when enabled (2-4 typical)
HOUGH_SAT_SCALE = 1
HOUGH_SAT_FLOOR = 5
HOUGH_SAT_CLAHE_ENABLED = False
HOUGH_SAT_CLAHE_CLIP = 2.5

# Preprocessing (Gaussian blur before Hough)
# GAUSSIAN_BLUR_KSIZE: (width, height), odd numbers; larger = smoother
# GAUSSIAN_BLUR_SIGMA: 0 = auto from ksize; higher = stronger blur
GAUSSIAN_BLUR_KSIZE = (9, 9)
GAUSSIAN_BLUR_SIGMA = 2.0

# HSV preprocessing -- CLAHE on V channel to equalize lighting before detection
# HSV_CLAHE_CLIP: higher = more contrast (2-4); HSV_CLAHE_TILE: (w, h) grid size
# HSV_SAT_SCALE: multiply saturation (1.0 = no change); >1 can help dim lighting
HSV_PREPROCESS_ENABLED = True
HSV_CLAHE_CLIP = 2.5
HSV_CLAHE_TILE = (8, 8)
HSV_SAT_SCALE = 1.15

# Arena ROI
# ARENA_ROI: single ROI fallback; (center_x_frac, center_y_frac, radius_frac).
# ARENA_ROI_HIGH: red, small, stadium center, high priority. Check first; if 2 beys found, done.
# ARENA_ROI_LOW: green, larger, low priority. Searched when < 2 beys in HIGH.
# Set ARENA_ROI_HIGH = None to disable dual-ROI and use ARENA_ROI only.
# PREFER_HIGH_PRIORITY: when full, replace an edge bey with unmatched center candidate
ARENA_ROI = (0.5, 0.52, 0.60)
PREFER_HIGH_PRIORITY = True
ARENA_ROI_HIGH = (0.5, 0.40, 0.40)
ARENA_ROI_LOW = (0.5, 0.52, 0.60)
ARENA_REFERENCE_DIR = "references/arena"
ARENA_RIM_SHRINK = 0.70
ARENA_ROI_OFFSET_X = 0
ARENA_ROI_OFFSET_Y = 0

# Tracking
# MAX_BEY_COUNT: max beys to track (2 = 1v1); discovery stops when reached
# MAX_RECOVERY_FRAMES: drop bey if not seen for this many frames
# BOOTSTRAP_MIN_SPEED: px/frame; below this a bey is "stationary"
# MAX_STUCK_FRAMES: if all beys stationary for this long, clear and rescan
MAX_BEY_COUNT = 2
MAX_RECOVERY_FRAMES = 25
BOOTSTRAP_MIN_SPEED = 8.0
MAX_STUCK_FRAMES = 20

# Position smoothing (1 = no smoothing; higher = smoother but laggier)
SMOOTH_WINDOW_SIZE = 1

# Scale detected circle radius for display
BEY_RADIUS_SCALE = 1.4
# Collision: margin extended from each bey circle; overlap of two zones = impact
# Minimum 1px used when 0. With -d, gray=circle, magenta=zone.
BEY_COLLISION_MARGIN_PX = 10

# Color sampling -- detects coloured bey chips against the white floor
# COLOR_SAMPLE_RADIUS_FRAC: fraction of circle radius to sample at center
# COLOR_SAT_MIN: minimum saturation (rejects white); lower = accept paler chips
# COLOR_CENTER_MIN_FILL: minimum fraction of center that must be saturated
# COLOR_MIN_HUE_SEPARATION: hue difference from rim to accept (rim rejection)
# REJECT_HUE_RANGES: [(lo, hi), ...] in OpenCV hue 0-180; exclude stadium elements (e.g. green rail)
# COLOR_HUE_TOLERANCE: max hue drift when adapting tracked color
# COLOR_ADAPT_RATE: blend rate toward observed hue (0 = no adaptation)
COLOR_SAMPLE_RADIUS_FRAC = 0.45
COLOR_SAT_MIN = 40
COLOR_VAL_MIN = 40
COLOR_CENTER_MIN_FILL = 0.10
COLOR_MIN_HUE_SEPARATION = 10
REJECT_HUE_RANGES = []   # Beyblade X stadium green X-rail; set [] if using green beys
# REJECT_NEAR_RIM_FRACTION: reject circles in outer X of arena (0 = disabled)
# Green rail sits at the rim; 0.08-0.12 rejects rail false positives, allows edge beys
REJECT_NEAR_RIM_FRACTION = 0.10

COLOR_HUE_TOLERANCE = 22
COLOR_ADAPT_RATE = 0.05

# Candidate matching (Hough circle -> tracked bey assignment)
# MATCH_MAX_DISTANCE: max px from Kalman prediction to accept as same bey
#   higher = tolerate faster motion / more missed frames
# MATCH_IDENTITY_WEIGHT: penalty for hue mismatch with stored identity; higher = avoid swapping bey1/bey2
# IDENTITY_HUE_MAX_DRIFT: max hue distance from identity to accept match; beyond = reject (0 = disabled)
# IDENTITY_BOOTSTRAP_FRAMES: collect hue samples over first N frames to stabilise identity (spin/incline)
MATCH_MAX_DISTANCE = 380
MATCH_IDENTITY_WEIGHT = 8.0
IDENTITY_HUE_MAX_DRIFT = 35
IDENTITY_BOOTSTRAP_FRAMES = 15

# New bey discovery
# DISCOVERY_MIN_SEPARATION: min px from any tracked bey to register new one
DISCOVERY_MIN_SEPARATION = 35

# Kalman filter
# KALMAN_PROCESS_NOISE: higher = trust Hough detections more than prediction
# KALMAN_MEASUREMENT_NOISE: lower = snap to detection faster
# KALMAN_LOSS_VELOCITY_DECAY: when Hough misses, velocity *= this each frame
#   higher = prediction keeps moving along last velocity
# KALMAN_MAX_PREDICTION_DRIFT: max px prediction can drift from last real position
# KALMAN_MAX_VELOCITY_PX: when prediction exceeds this velocity (px/frame), reject it and hold
#   position; rely on Hough to re-acquire the colored circle (0 = no limit)
# Stadium boundary: when bey is near the rim, remove outward velocity component (0.92 = outer 8%)
KALMAN_RIM_CLAMP_FRAC = 1
# Circular motion: fit recent positions to circle, predict along arc (Beyblades orbit stadium)
# CIRCULAR_PREDICTION_ENABLED: use circular model when orbit is detectable
# CIRCULAR_HISTORY_LEN: positions to keep for circle fit (need 5+ for reliable fit)
KALMAN_ENABLED = True
KALMAN_PROCESS_NOISE = 0.45
CIRCULAR_PREDICTION_ENABLED = True
CIRCULAR_HISTORY_LEN = 20
KALMAN_MEASUREMENT_NOISE = 0.15
KALMAN_LOSS_VELOCITY_DECAY = 0.92
KALMAN_MAX_PREDICTION_DRIFT = 100
KALMAN_MAX_VELOCITY_PX = 55

# Collision detection
# COLLISION_COOLDOWN_FRAMES: min frames between collision events
# COLLISION_MIN_APPROACH_SPEED: min combined approach speed (px/frame) to trigger
# COLLISION_KALMAN_CONFIRM: when True, require velocity direction change vs Kalman
#   prediction (reflection) to confirm impact; reduces false positives
# COLLISION_VELOCITY_REVERSAL_MIN_SPEED: min speed (px/frame) to check direction
# COLLISION_VELOCITY_REVERSAL_COS: cos(angle) threshold; < 0.5 = > 60 deg deflection
COLLISION_COOLDOWN_FRAMES = 5
COLLISION_MIN_APPROACH_SPEED = 20.0
COLLISION_KALMAN_CONFIRM = False
COLLISION_VELOCITY_REVERSAL_MIN_SPEED = 5.0
COLLISION_VELOCITY_REVERSAL_COS = 0.5

# Debug
DEBUG_PRINT = False   # True = print collision/recovery events to console

# Overlay colors (BGR)
COLOR_BEY_1 = (0, 255, 0)
COLOR_BEY_2 = (255, 0, 0)
COLOR_VELOCITY = (0, 255, 255)
COLOR_IMPACT = (0, 0, 255)
FONT_SCALE = 0.6
FONT_THICKNESS = 2

# Trail effect (-e/--effect)
TRAIL_MAX_LEN = 100
TRAIL_BASE_RADIUS = 6
TRAIL_POINTS_PER_SEGMENT = 20

# Impact effect (-e/--effect)
IMPACT_FLASH_FRAMES = 9
IMPACT_FLASH_MAX_RADIUS = 55
IMPACT_FLASH_COLOR = (0, 200, 255)
# Force-based scaling: heavier impacts = larger flash
IMPACT_FORCE_RADIUS_SCALE = 0.002   # extra radius per unit force above baseline
IMPACT_MIN_FORCE_BASELINE = 100.0  # force below this uses base radius only
IMPACT_RIPPLE_RINGS = 3            # concentric expanding rings
IMPACT_LABEL_ENABLED = True        # show IMPACT! text at hit point
IMPACT_LABEL_SHOW_FORCE = False    # append force value to label

# Wall collision
WALL_COLLISION_MARGIN = 5
WALL_FLASH_FRAMES = 6
WALL_FLASH_COLOR = (255, 180, 0)

# Web SFX output (--web)
WEB_WS_HOST = "127.0.0.1"
WEB_WS_PORT = 8765

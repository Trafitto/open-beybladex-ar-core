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
# HOUGH_MIN_DIST: minimum px between circle centers; lower = 2 beys when overlapping
# HOUGH_PARAM1: Canny edge high threshold; HOUGH_PARAM2: accumulator threshold
#   lower param2 = more detections but noisier; lower (~14-16) helps when beys blur
# HOUGH_MIN_RADIUS, HOUGH_MAX_RADIUS: px range for bey circle size
HOUGH_DP = 1.2
HOUGH_MIN_DIST = 18
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
#   Higher helps muted/dark beys stand out vs white floor; rail may clip at 255
# HOUGH_SAT_FLOOR: clip values below this to 0 (suppress near-white noise, 0 = disabled)
# HOUGH_SAT_CLAHE_ENABLED: apply CLAHE to S for uneven lighting; helps muted beys
# HOUGH_SAT_CLAHE_CLIP: CLAHE clip limit when enabled (2-4 typical)
HOUGH_SAT_SCALE = 1.35
HOUGH_SAT_FLOOR = 3
HOUGH_SAT_CLAHE_ENABLED = True
HOUGH_SAT_CLAHE_CLIP = 2.8

# Preprocessing (Gaussian blur before Hough)
# GAUSSIAN_BLUR_KSIZE: (width, height), odd numbers; larger = smoother
# GAUSSIAN_BLUR_SIGMA: 0 = auto from ksize; higher = stronger blur
GAUSSIAN_BLUR_KSIZE = (9, 9)
GAUSSIAN_BLUR_SIGMA = 2.0

# HSV preprocessing -- CLAHE on V channel to equalize lighting before detection
# Helps with dome shadows and uneven illumination
# HSV_CLAHE_CLIP: higher = more contrast (2-4); HSV_CLAHE_TILE: (w, h) grid size
# HSV_SAT_SCALE: multiply saturation (1.0 = no change); >1 helps muted/dark beys
HSV_PREPROCESS_ENABLED = True
HSV_CLAHE_CLIP = 2.5
HSV_CLAHE_TILE = (8, 8)
HSV_SAT_SCALE = 1.35

# Arena ROI
# ARENA_ROI: single ROI fallback; (center_x_frac, center_y_frac, radius_frac).
# ARENA_ROI_HIGH: red circle, high-priority zone (center). Prefer tracking here.
# ARENA_ROI_LOW: set to None for red circle only (no green)
# PREFER_HIGH_PRIORITY: prefer candidates in red zone when matching
ARENA_ROI = (0.5, 0.52, 0.60)
PREFER_HIGH_PRIORITY = True
ARENA_ROI_HIGH = (0.5, 0.40, 0.40)
ARENA_ROI_LOW = None
ARENA_REFERENCE_DIR = "references/arena"
ARENA_RIM_SHRINK = 0.70
ARENA_ROI_OFFSET_X = 0
ARENA_ROI_OFFSET_Y = 0

# Tracking
# MAX_BEY_COUNT: max beys to track (2 = both in arena)
# MAX_RECOVERY_FRAMES: drop bey if not seen for this many frames (higher = keep rail-accelerating bey longer)
MAX_BEY_COUNT = 2
MAX_RECOVERY_FRAMES = 20
BOOTSTRAP_MIN_SPEED = 8.0
MAX_STUCK_FRAMES = 10

# Zero velocity = wrong object (e.g. tracking static green rail)
# Drop bey if speed < threshold for this many consecutive frames (0 = disabled)
ZERO_VELOCITY_CLEAR_FRAMES = 30
ZERO_VELOCITY_THRESHOLD = 4.0

# Position smoothing (1 = no smoothing; higher = smoother but laggier)
SMOOTH_WINDOW_SIZE = 1

# Scale detected circle radius for display
BEY_RADIUS_SCALE = 1.4
# Beyblade identification: custom labels per slot (first bey=0, second=1)
# Empty or ["", ""] = use color name from hue (Red, Blue, etc.)
BEY_IDENTITIES = ["Beyblade 1", "Beyblade 2"]   # e.g. ["Player 1", "Player 2"] or ["Dragon", "Phoenix"]
# Collision: margin extended from each bey circle; overlap of two zones = impact
# Minimum 1px used when 0. With -d, gray=circle, magenta=zone.
BEY_COLLISION_MARGIN_PX = 10

# Color sampling -- detects coloured bey chips against the white floor
# COLOR_SAMPLE_RADIUS_FRAC: fraction of circle radius to sample at center
# COLOR_SAT_MIN: minimum saturation (rejects white); lower = accept paler chips
# COLOR_CENTER_MIN_FILL: minimum fraction of center that must be saturated
# COLOR_MIN_HUE_SEPARATION: reject circles whose center hue is within this of rail hue
# REJECT_HUE_RANGES: [(lo, hi), ...] in OpenCV hue 0-180; exclude stadium elements (e.g. green rail)
# COLOR_HUE_TOLERANCE: max hue drift when adapting tracked color
# COLOR_ADAPT_RATE: blend rate toward observed hue (0 = no adaptation)
COLOR_SAMPLE_RADIUS_FRAC = 0.45
COLOR_SAT_MIN = 32
COLOR_VAL_MIN = 35
COLOR_CENTER_MIN_FILL = 0.08
COLOR_MIN_HUE_SEPARATION = 10
REJECT_HUE_RANGES = []   # Beyblade X stadium green X-rail; set [] if using green beys
# REJECT_NEAR_RIM_FRACTION: reject circles in outer X of arena (0 = disabled)
# Green rail sits at the rim; 0.12-0.15 rejects rail false positives
REJECT_NEAR_RIM_FRACTION = 0.15

# Rail mask: reduce saturation in green rail region before Hough (stadium is static)
# RAIL_MASK_ENABLED: build mask from first frame, zero out S in rail area
# RAIL_MASK_HUE_LO/HI: hue range for green rail (OpenCV 0-180); 35-95 typical
# RAIL_MASK_SAT_MIN: min saturation to count as rail
# RAIL_MASK_ANNULUS_INNER: inner radius frac (0.80 = outer 20%, narrower band)
# RAIL_MASK_OUTER_SCALE: extend beyond arena (1.05 = barely past rim)
RAIL_MASK_ENABLED = True
RAIL_MASK_HUE_LO = 30
RAIL_MASK_HUE_HI = 100
RAIL_MASK_SAT_MIN = 45
RAIL_MASK_ANNULUS_INNER = 0.70
RAIL_MASK_OUTER_SCALE = 0.75
RAIL_MASK_SAVE_PATH = ""   # e.g. "output/rail_mask.png" to save mask for inspection
# RAIL_MASK_POINTS_FILE: save/load polygon points (tracking area); used when -rm or file missing
RAIL_MASK_POINTS_FILE = "output/rail_mask_points.json"
# POLYGON_EDGE_MARGIN: reject circles within this many px of polygon edge (0 = disabled)
POLYGON_EDGE_MARGIN = 18
# RED_ZONE_POINTS_FILE: save/load red zone from -rz selection; used when -rz or file exists
RED_ZONE_POINTS_FILE = "output/red_zone.json"
# RAIL_TRACKING_ALLOW_EDGE: when True, allow near-edge circles if within MATCH_MAX_DISTANCE of a tracked bey
# Enables tracking when bey accelerates along the rail (attack launch)
RAIL_TRACKING_ALLOW_EDGE = True

# Plastic dome cover: reduces glare/specular reflections and optional text-region exclude
# DOME_GLARE: zero S in bright specular spots (high V, low S) before Hough
DOME_GLARE_ENABLED = True
DOME_GLARE_V_MIN = 200    # V above this = potential glare (lower = catch more)
DOME_GLARE_S_MAX = 55     # if S below this in bright region, treat as specular
# DOME_EXCLUDE_WEDGE: optional wedge (degrees) to exclude where "Beyblade X" text is
# Angles from top (0), clockwise: 90=right, 180=bottom, 270=left. Set to (-1,-1) to disable
DOME_EXCLUDE_WEDGE_ENABLED = False
DOME_EXCLUDE_ANGLE_START = 135   # e.g. exclude bottom-left quadrant
DOME_EXCLUDE_ANGLE_END = 225
# DOME_MASK_SAVE_PATH: save path for run_dome_mask_snapshot.py (e.g. output/dome_mask.png)
DOME_MASK_SAVE_PATH = "output/dome_mask.png"

# Hand exclusion: when hands appear (e.g. during launch), exclude them from Beyblade tracking
# Uses MediaPipe Hands to detect hands and mask them out before Hough circle detection
# HAND_EXCLUSION_ENABLED: enable hand exclusion (requires mediapipe)
# HAND_MASK_DILATE_PX: expand hand mask by this many pixels (landmarks are on joints, not outline)
# HAND_MIN_DETECTION_CONFIDENCE: MediaPipe min confidence for hand detection (0-1)
# HAND_OVERLAP_REJECT_POINTS: reject circle if this many sample points (center + 8 on perimeter) fall in hand
#   Beyblade held in hand = circle overlaps fingers; 3+ points catches "show-off" phase before launch
HAND_EXCLUSION_ENABLED = True
HAND_MASK_DILATE_PX = 50
HAND_MIN_DETECTION_CONFIDENCE = 0.5
HAND_OVERLAP_REJECT_POINTS = 5

# Stadium contact: reject circles whose surrounding context does not look like stadium floor
# When Beyblade is ON the stadium, the ring around it is white/light (low S). When held, it's hand/launcher.
# STADIUM_CONTACT_ENABLED: require surrounding pixels to look like stadium floor
# STADIUM_CONTACT_RING_INNER: inner radius of annulus as multiple of circle radius (1.05 = just outside)
# STADIUM_CONTACT_RING_OUTER: outer radius of annulus
# STADIUM_CONTACT_MAX_SAT: pixels with S below this count as "stadium floor"; white floor = low S
# STADIUM_CONTACT_MIN_FRAC: min fraction of ring pixels that must be stadium-like (0-1)
# STADIUM_CONTACT_SKIP_NEAR_TRACKED: when True, skip check if candidate is near an existing tracked bey
#   that is moving fast (rail acceleration: bey moves to rim where ring may include green rail)
# STADIUM_CONTACT_SKIP_MIN_SPEED: only skip when nearby bey has speed >= this (px/frame); avoids
#   feedback loop where held bey keeps passing because we keep updating it
# STADIUM_CONTACT_MAX_SKIN_FRAC: reject if this fraction of ring is skin-colored (hand present); 0=disabled
# STADIUM_CONTACT_REQUIRE_LOWER_HALF: when True, also require the lower half of ring to be stadium-like
#   (Beyblade touches floor below it; when held, lower half = hand)
STADIUM_CONTACT_ENABLED = True
STADIUM_CONTACT_RING_INNER = 1.05
STADIUM_CONTACT_RING_OUTER = 2.0
STADIUM_CONTACT_MAX_SAT = 58
STADIUM_CONTACT_MIN_FRAC = 0.35
STADIUM_CONTACT_SKIP_NEAR_TRACKED = True
STADIUM_CONTACT_SKIP_MIN_SPEED = 4.0
STADIUM_CONTACT_MAX_SKIN_FRAC = 0.25
STADIUM_CONTACT_REQUIRE_LOWER_HALF = False

COLOR_HUE_TOLERANCE = 22
COLOR_ADAPT_RATE = 0.05

# Candidate matching (Hough circle -> tracked bey assignment)
# MATCH_MAX_DISTANCE: max px from Kalman prediction to accept as same bey
#   higher = tolerate faster motion / more missed frames (rail acceleration)
# MATCH_MAX_DISTANCE_FAST: when bey speed > MATCH_FAST_SPEED_THRESHOLD, use this (0 = use MATCH_MAX_DISTANCE only)
# MATCH_IDENTITY_WEIGHT: penalty for hue mismatch with stored identity; higher = avoid swapping bey1/bey2
# IDENTITY_HUE_MAX_DRIFT: max hue distance from identity to accept match; beyond = reject (0 = disabled)
# IDENTITY_BOOTSTRAP_FRAMES: collect hue samples over first N frames to stabilise identity (spin/incline)
MATCH_MAX_DISTANCE = 500
MATCH_MAX_DISTANCE_FAST = 550
MATCH_FAST_SPEED_THRESHOLD = 40
MATCH_IDENTITY_WEIGHT = 8.0
IDENTITY_HUE_MAX_DRIFT = 35
IDENTITY_BOOTSTRAP_FRAMES = 15

# Launch-phase clear: when all beys have low speed for N frames, clear tracking.
# Show-off/launch = held beys (low speed). Battle = spinning (high speed).
# LAUNCH_CLEAR_ENABLED: enable this rule
# LAUNCH_CLEAR_REQUIRE_HANDS: also require hands detected (stricter; skip when MediaPipe fails)
# LAUNCH_CLEAR_SPEED: max speed (px/frame) to count as "launch phase"
# LAUNCH_CLEAR_FRAMES: consecutive frames before clearing
LAUNCH_CLEAR_ENABLED = True
LAUNCH_CLEAR_REQUIRE_HANDS = False
LAUNCH_CLEAR_SPEED = 8.0
LAUNCH_CLEAR_FRAMES = 10

# New bey discovery
# DISCOVERY_MIN_SEPARATION: min px from any tracked bey to register new one
# DISCOVERY_REQUIRE_MOTION: only register new bey if frame-diff shows motion at that spot
#   Held Beyblades are static; spinning ones have motion. Helps exclude show-off phase.
DISCOVERY_MIN_SEPARATION = 35
DISCOVERY_REQUIRE_MOTION = True
# CANDIDATE_MIN_SEPARATION: two Hough circles closer than this are same object; keep one
CANDIDATE_MIN_SEPARATION = 28

# Kalman filter
# KALMAN_PROCESS_NOISE: higher = trust Hough detections more than prediction
# KALMAN_MEASUREMENT_NOISE: lower = snap to detection faster
# KALMAN_LOSS_VELOCITY_DECAY: when Hough misses, velocity *= this each frame
#   higher = prediction keeps moving along last velocity
# KALMAN_MAX_PREDICTION_DRIFT: max px prediction can drift from last real position (higher = track fast rail acceleration)
# KALMAN_MAX_VELOCITY_PX: when prediction exceeds this velocity (px/frame), reject and hold (0 = no limit; use 0 for rail)
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
KALMAN_LOSS_VELOCITY_DECAY = 0.95
KALMAN_MAX_PREDICTION_DRIFT = 200
KALMAN_MAX_VELOCITY_PX = 120

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
DEBUG_HIDE_RED_CIRCLE_WHEN_POLYGON = False  # Keep red circle visible (high-priority zone)

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

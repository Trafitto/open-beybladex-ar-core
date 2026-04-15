"""
Configuration for Beyblade X arena tracking.

Pipeline: Hough circles -> center color sample -> motion check -> Kalman track.

Tuned for: PS3 Eye camera (OV534 + OV772x, gspca_ov534 driver),
           2 beyblades, very fast motion, white stadium background, steady camera.
"""

# Camera
CAMERA_INDEX = 4         # Webcam device index (0 = default)
TARGET_FPS = 60          # PS3 Eye max at 640x480

# ---------------------------------------------------------------------------
# PS3 Eye hardware controls (applied automatically via v4l2-ctl at startup)
# ---------------------------------------------------------------------------
# OV7725 sensor: 6µm pixels, 50dB SNR, 60dB dynamic range, 3.0 V/(Lux·sec)
# Set PS3EYE_ENABLED = False to skip hardware configuration entirely.
# Requires v4l2-utils: sudo apt install v4l2-utils
#
# For fast-object tracking the priorities are:
#   1) LOW exposure -> minimise motion blur (spinning beys smear at high exposure)
#   2) Moderate gain -> compensate brightness without excessive noise
#   3) Auto OFF for everything -> stable frame-to-frame brightness for thresholds
PS3EYE_ENABLED = True
PS3EYE_AUTOGAIN = 0           # 0 = manual gain+exposure, 1 = auto
PS3EYE_AUTO_EXPOSURE = 1      # v4l2 menu: 0 = Auto Mode, 1 = Manual Mode
PS3EYE_AUTO_WHITE_BALANCE = 0  # 0 = manual, 1 = auto
PS3EYE_EXPOSURE = 40          # 0-255 (low = less motion blur, darker image)
PS3EYE_GAIN = 30              # 0-63  (compensates low exposure; higher = noisier)
PS3EYE_BRIGHTNESS = 128       # 0-255 (sensor-level, before capture)
PS3EYE_CONTRAST = 128         # 0-255 (default 32; moderate boost helps bey/floor sep)
PS3EYE_SATURATION_HW = 80     # 0-255 (sensor-level saturation; boost helps color sampling)
PS3EYE_SHARPNESS = 32         # 0-63  (edge enhancement; too high amplifies noise)
PS3EYE_RED_BALANCE = 128      # 0-255 (only when AWB is off)
PS3EYE_BLUE_BALANCE = 128     # 0-255 (only when AWB is off)
PS3EYE_HFLIP = 0
PS3EYE_VFLIP = 0
PS3EYE_POWER_LINE_FREQ = 1    # 0 = disabled, 1 = 50Hz (EU/most), 2 = 60Hz (US)

# Physical stadium dimensions (for pixel -> real-world mapping)
# Beyblade X Xtreme Stadium outer diameter ~= 400 mm.
# mm_per_pixel is computed at runtime from the detected arena ROI radius.
STADIUM_DIAMETER_MM = 400.0
# Tolerance (in mm) from the stadium wall to count as a wall hit.
# Roughly one beyblade width.
WALL_HIT_TOLERANCE_MM = 15.0

# Processing resolution -- 640x480 is the PS3 Eye native VGA mode.
# Using native avoids resampling artefacts from the noisy sensor.
PROCESS_WIDTH = 640
PROCESS_HEIGHT = 480

# Hough Circle detection (runs every frame to find bey candidates)
# HOUGH_DP: inverse ratio accumulator/image resolution (1.0-2.0 typical)
# HOUGH_MIN_DIST: minimum px between circle centers; lower = 2 beys when overlapping
# HOUGH_PARAM1: Canny edge high threshold; HOUGH_PARAM2: accumulator threshold
#   lower param2 = more detections but noisier; lower (~14-16) helps when beys blur
# HOUGH_MIN_RADIUS, HOUGH_MAX_RADIUS: px range for bey circle size
# PS3 Eye: raised param1/param2 vs generic webcam to reject sensor noise.
HOUGH_DP = 1.2
HOUGH_MIN_DIST = 20
HOUGH_PARAM1 = 90
HOUGH_PARAM2 = 14
HOUGH_MIN_RADIUS = 3
HOUGH_MAX_RADIUS = 12

# Detection method: "hough" (HoughCircles) or "contour" (findContours)
# Contour is significantly faster (~3-5x) and works well when beys are
# saturated blobs on a white floor.  Hough is more shape-specific.
DETECTION_METHOD = "contour"

# Detection input channel: "combined", "saturation", or "grayscale"
# Combined: max(inverted_grayscale, boosted_saturation) -- a bey visible in
#   EITHER channel gets detected. Best overall for mixed bey types.
# Grayscale (inverted): dark beys on white floor, color-blind.
# Saturation: only works when beys have vivid colour.
HOUGH_DETECTION_CHANNEL = "combined"

# Saturation channel tuning (only when HOUGH_DETECTION_CHANNEL == "saturation")
# HOUGH_SAT_SCALE: multiply S channel (1.0 = no change); >1 boosts colored vs white
#   Higher helps muted/dark beys stand out vs white floor; rail may clip at 255
# HOUGH_SAT_FLOOR: clip values below this to 0 (suppress near-white noise, 0 = disabled)
# HOUGH_SAT_CLAHE_ENABLED: apply CLAHE to S for uneven lighting; helps muted beys
# HOUGH_SAT_CLAHE_CLIP: CLAHE clip limit when enabled (2-4 typical)
# OV7725: floor=70 suppresses chroma noise on neutral surfaces at gain=30;
#   CLAHE clip=2.2 avoids amplifying sensor noise (50dB SNR).
HOUGH_SAT_SCALE = 1.0
HOUGH_SAT_FLOOR = 70
# SAT_BOOST: multiplicative boost on inv-gray where saturation exists.
# Colored bey chips get amplified; neutral floor stays unchanged.
SAT_BOOST = 1.4
HOUGH_SAT_CLAHE_ENABLED = True
HOUGH_SAT_CLAHE_CLIP = 2.2

# Preprocessing (Gaussian blur before detection)
# OV7725 at low exposure + moderate gain produces visible pixel noise.
# (9,9) kernel with sigma 2.0 smooths noise without killing the ~20px bey blobs.
GAUSSIAN_BLUR_KSIZE = (9, 9)
GAUSSIAN_BLUR_SIGMA = 2.0

# Contour detection (only when DETECTION_METHOD == "contour")
# For grayscale channel: the image is inverted (dark beys become bright blobs).
#   CONTOUR_THRESHOLD sets the brightness cutoff on the inverted image.
#   Bey body (orig ~40-120) -> inverted ~135-215. Floor (orig ~200+) -> inverted ~15-55.
#   Threshold ~100 separates them cleanly.
# For saturation channel: threshold on S values (colored vs white).
# CONTOUR_MORPH_KSIZE: kernel size for morphological open/close (odd number)
# CONTOUR_MIN_AREA / MAX_AREA: filter contours by pixel area (px^2)
# CONTOUR_MIN_CIRCULARITY: 4*pi*area/perimeter^2 (1.0 = perfect circle)
# With inv-gray as primary signal: floor inverts to ~5-30, beys to ~100-200.
# Threshold 40 catches beys cleanly without floor noise.
CONTOUR_THRESHOLD = 60
CONTOUR_MORPH_KSIZE = 5
CONTOUR_MIN_AREA = 80
CONTOUR_MAX_AREA = 8000
CONTOUR_MIN_CIRCULARITY = 0.20
# Erode the play area inward by this many px to mask the inner rail edge.
# The dark green rail inverts to bright in grayscale mode and merges with bey blobs.
CONTOUR_RIM_ERODE = 5

# Adaptive thresholding: local contrast detection for faint objects.
# OR'd with global threshold -- catches beys that are only slightly
# different from the surrounding floor (like the orange metallic bey).
# ADAPTIVE_THRESH_BLOCK: neighbourhood size for local mean (must be odd, ~31-71)
# ADAPTIVE_THRESH_C: negative = pixel must be C brighter than local mean to pass
#   More negative = stricter (fewer false positives); less negative = catches fainter objects
ADAPTIVE_THRESH_ENABLED = False
ADAPTIVE_THRESH_BLOCK = 51
ADAPTIVE_THRESH_C = -20

# Edge detection: Canny edges dilated into blobs, added as 3rd signal in combined mode.
# Catches bey outlines even when the body brightness matches the floor.
# EDGE_CANNY_LOW/HIGH: Canny hysteresis thresholds
# EDGE_DILATE: dilate edges into filled blobs (px radius)
# EDGE_BOOST: multiplicative boost applied to inv+sat where edges exist (>1 = amplify)
#   Edges amplify existing signal at object boundaries rather than adding new signal,
#   preventing floor edges from creating false blobs.
EDGE_DETECT_ENABLED = True
EDGE_CANNY_LOW = 50
EDGE_CANNY_HIGH = 150
EDGE_DILATE = 3
EDGE_BOOST = 1.5

# Background subtraction (MOG2): learns what the static arena looks like and
# highlights anything different (= beys). Very robust regardless of bey color.
# BG_SUB_HISTORY: frames for background model (300 @ 60fps = 5 seconds)
# BG_SUB_VAR_THRESHOLD: pixel variance to consider foreground (lower = more sensitive)
# BG_SUB_LEARNING_RATE: -1 = auto, 0 = don't learn, 0.001-0.01 = slow learning
# BG_SUB_BLUR: blur the foreground mask to fill gaps (0 = disabled)
# BG_SUB_BOOST: multiply inv+sat where MOG2 says foreground (>1 = amplify beys)
# BG_SUB_PENALTY: multiply inv+sat where MOG2 says background (<1 = suppress static noise)
BG_SUB_ENABLED = False
BG_SUB_HISTORY = 300
# Higher variance threshold tolerates OV7725 sensor noise at gain=30
BG_SUB_VAR_THRESHOLD = 50
BG_SUB_LEARNING_RATE = 0.002
BG_SUB_BLUR = 7
# Mild penalty: don't suppress the inv-gray signal too aggressively
# in "background" areas -- the bey IS the signal.
BG_SUB_BOOST = 1.3
BG_SUB_PENALTY = 0.85

# Motion mask: frame differencing to boost moving objects and suppress static noise.
# Moving pixels (beys spinning/translating) get their combined signal boosted;
# static pixels (screws, floor texture, table) get penalised.
# MOTION_MASK_THRESHOLD: pixel diff above this = motion (0-255)
# MOTION_MASK_DILATE: dilate the motion region by this many px (covers full bey body)
# MOTION_MASK_BOOST: multiply signal in moving areas (>1 = amplify)
# MOTION_MASK_STATIC_PENALTY: multiply signal in static areas (<1 = suppress)
MOTION_MASK_ENABLED = True
# Higher than typical (25 vs 10-15) to not trigger on OV7725 sensor noise
MOTION_MASK_THRESHOLD = 25
MOTION_MASK_DILATE = 10
# Gentle boost/penalty: beys spinning in place still need detection,
# so static penalty must be close to 1.0.
MOTION_MASK_BOOST = 1.3
MOTION_MASK_STATIC_PENALTY = 0.6

# Dynamic ROI: crop a small window around each tracked bey's predicted position
# instead of scanning the full frame. Full-frame detection is used as fallback
# when ROI search fails or when discovering new beys.
# ROI_ENABLED: master switch (False = always full-frame, for debugging)
# ROI_SIZE: side length in pixels of the square crop (0 = disabled)
# ROI_FALLBACK_FRAMES: after this many consecutive ROI misses, trigger full-frame
ROI_ENABLED = True
ROI_SIZE = 200
ROI_FALLBACK_FRAMES = 1

# HSV preprocessing -- CLAHE on V channel to equalize lighting before detection
# Helps with dome shadows and uneven illumination
# HSV_CLAHE_CLIP: higher = more contrast (2-4); HSV_CLAHE_TILE: (w, h) grid size
# HSV_SAT_SCALE: multiply saturation (1.0 = no change); >1 helps muted/dark beys
# PS3 Eye: gentler CLAHE clip (2.0) to avoid boosting OV772x noise; slightly
#   higher sat scale (1.4) compensates for the sensor's weaker colour response
#   at low exposure settings.
HSV_PREPROCESS_ENABLED = True
HSV_CLAHE_CLIP = 2.0
HSV_CLAHE_TILE = (8, 8)
# Reduced from 1.8 because hardware saturation (PS3EYE_SATURATION_HW=80) already
# boosts at the sensor level. Combined effect: 80/64 * 1.4 = ~1.75x total.
HSV_SAT_SCALE = 1.4

# Arena ROI
# ARENA_ROI: single ROI fallback; (center_x_frac, center_y_frac, radius_frac).
# ARENA_ROI_HIGH: red circle, high-priority zone (center). Prefer tracking here.
# ARENA_ROI_LOW: set to None for red circle only (no green)
# PREFER_HIGH_PRIORITY: prefer candidates in red zone when matching
ARENA_ROI = (0.5, 0.52, 0.60)
PREFER_HIGH_PRIORITY = True
ARENA_ROI_HIGH = (0.5, 0.40, 0.35)
ARENA_ROI_LOW = None
ARENA_REFERENCE_DIR = "references/arena"
ARENA_RIM_SHRINK = 0.70
ARENA_ROI_OFFSET_X = 0
ARENA_ROI_OFFSET_Y = 0

# Tracking
# MAX_BEY_COUNT: max beys to track (2 = both in arena)
# MAX_RECOVERY_FRAMES: drop bey if not seen for this many frames (higher = keep rail-accelerating bey longer)
MAX_BEY_COUNT = 2
MAX_RECOVERY_FRAMES = 30
BOOTSTRAP_MIN_SPEED = 6.0
MAX_STUCK_FRAMES = 15

# Zero velocity = wrong object (e.g. tracking static green rail)
# Drop bey if speed < threshold for this many consecutive frames (0 = disabled)
ZERO_VELOCITY_CLEAR_FRAMES = 15
ZERO_VELOCITY_THRESHOLD = 4.0

# Position smoothing (1 = no smoothing; higher = smoother but laggier)
SMOOTH_WINDOW_SIZE = 1

# Physical blade radius in pixels (used for collision and overlay).
# The tracker detects the small center chip; this constant represents
# the full blade circle for physics and drawing.
BEY_BLADE_RADIUS_PX = 17

# Scale detected circle radius for display
BEY_RADIUS_SCALE = 1.4
# Beyblade identification: custom labels per slot (first bey=0, second=1)
# Empty or ["", ""] = use color name from hue (Red, Blue, etc.)
BEY_IDENTITIES = ["Blader 1", "Blader 2"]   # e.g. ["Player 1", "Player 2"] or ["Dragon", "Phoenix"]
# Collision: margin extended from each bey circle; overlap of two zones = impact
# Minimum 1px used when 0. With -d, gray=circle, magenta=zone.
BEY_COLLISION_MARGIN_PX = 5

# Color sampling -- detects coloured bey chips against the white floor
# COLOR_SAMPLE_RADIUS_FRAC: fraction of circle radius to sample at center
# COLOR_SAT_MIN: minimum saturation (rejects white); lower = accept paler chips
# COLOR_CENTER_MIN_FILL: minimum fraction of center that must be saturated
# COLOR_MIN_HUE_SEPARATION: reject circles whose center hue is within this of rail hue
# REJECT_HUE_RANGES: [(lo, hi), ...] in OpenCV hue 0-180; exclude stadium elements (e.g. green rail)
# COLOR_HUE_TOLERANCE: max hue drift when adapting tracked color
# COLOR_ADAPT_RATE: blend rate toward observed hue (0 = no adaptation)
# PS3 Eye: lower COLOR_SAT_MIN (28) because the sensor's colour response is
#   weaker at low exposure; lower COLOR_VAL_MIN (30) to catch darker beys on
#   the dimmer image; slightly relaxed fill (0.06) for the noisier centre patch.
COLOR_SAMPLE_RADIUS_FRAC = 0.40
COLOR_SAT_MIN = 15
COLOR_VAL_MIN = 20
COLOR_CENTER_MIN_FILL = 0.05
COLOR_MIN_HUE_SEPARATION = 12
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
# PS3 Eye: lowered sat min (35) -- the sensor under-saturates greens at low
#   exposure, so the rail region needs a more permissive threshold to be caught.
RAIL_MASK_ENABLED = True
RAIL_MASK_HUE_LO = 30
RAIL_MASK_HUE_HI = 100
RAIL_MASK_SAT_MIN = 35
RAIL_MASK_ANNULUS_INNER = 0.70
RAIL_MASK_OUTER_SCALE = 0.75
RAIL_MASK_SAVE_PATH = ""   # e.g. "output/rail_mask.png" to save mask for inspection
# RAIL_MASK_POINTS_FILE: save/load polygon points (tracking area); used when -rm or file missing
RAIL_MASK_POINTS_FILE = "output/rail_mask_points.json"
# POLYGON_EDGE_MARGIN: reject circles within this many px of polygon edge (0 = disabled)
POLYGON_EDGE_MARGIN = 8
# RED_ZONE_POINTS_FILE: save/load red zone from -rz selection; used when -rz or file exists
RED_ZONE_POINTS_FILE = "output/red_zone.json"
# RAIL_TRACKING_ALLOW_EDGE: when True, allow near-edge circles if within MATCH_MAX_DISTANCE of a tracked bey
# Enables tracking when bey accelerates along the rail (attack launch)
RAIL_TRACKING_ALLOW_EDGE = True

# Plastic dome cover: reduces glare/specular reflections and optional text-region exclude
# DOME_GLARE: zero S in bright specular spots (high V, low S) before Hough
# PS3 Eye: lower V_MIN (185) because the sensor's dynamic range is narrower --
#   glare peaks lower than on modern webcams; raised S_MAX (65) to catch
#   slightly-coloured specular patches the OV772x can produce.
DOME_GLARE_ENABLED = True
DOME_GLARE_V_MIN = 185    # V above this = potential glare (lower = catch more)
DOME_GLARE_S_MAX = 65     # if S below this in bright region, treat as specular
# DOME_EXCLUDE_WEDGE: optional wedge (degrees) to exclude where "Beyblade X" text is
# Angles from top (0), clockwise: 90=right, 180=bottom, 270=left. Set to (-1,-1) to disable
DOME_EXCLUDE_WEDGE_ENABLED = False
DOME_EXCLUDE_ANGLE_START = 135   # e.g. exclude bottom-left quadrant
DOME_EXCLUDE_ANGLE_END = 225
# DOME_MASK_SAVE_PATH: save path for run_dome_mask_snapshot.py (e.g. output/dome_mask.png)
DOME_MASK_SAVE_PATH = "output/dome_mask.png"

# PS3 Eye: wider hue tolerance (25) to accommodate the noisier hue readings
#   from the OV772x; slightly faster adapt (0.08) to track the jittery hue.
COLOR_HUE_TOLERANCE = 25
COLOR_ADAPT_RATE = 0.08

# Candidate matching (Hough circle -> tracked bey assignment)
# MATCH_MAX_DISTANCE: max px from Kalman prediction to accept as same bey
#   higher = tolerate faster motion / more missed frames (rail acceleration)
# MATCH_MAX_DISTANCE_FAST: when bey speed > MATCH_FAST_SPEED_THRESHOLD, use this (0 = use MATCH_MAX_DISTANCE only)
# MATCH_IDENTITY_WEIGHT: penalty for hue mismatch with stored identity; higher = avoid swapping bey1/bey2
# IDENTITY_HUE_MAX_DRIFT: max hue distance from identity to accept match; beyond = reject (0 = disabled)
# IDENTITY_BOOTSTRAP_FRAMES: collect hue samples over first N frames to stabilise identity (spin/incline)
# PS3 Eye: wider hue drift (40) to handle OV772x chroma noise;
#   longer bootstrap (20) gives more frames to average out noisy hue samples.
# At 60fps a bey crosses the ~340px arena in ~1s = ~6px/frame typical,
# ~30px/frame max during rail acceleration.  Distance beyond ~120px
# means the prediction is wrong and the candidate is a different object.
MATCH_MAX_DISTANCE = 150
MATCH_MAX_DISTANCE_FAST = 200
MATCH_MAX_ACCELERATION = 5000.0
MATCH_FAST_SPEED_THRESHOLD = 40
MATCH_IDENTITY_WEIGHT = 8.0
IDENTITY_HUE_MAX_DRIFT = 40
IDENTITY_BOOTSTRAP_FRAMES = 20

# New bey discovery
# DISCOVERY_MIN_SEPARATION: min px from any tracked bey to register new one
DISCOVERY_MIN_SEPARATION = 35
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
KALMAN_LOSS_VELOCITY_DECAY = 0.5
KALMAN_MAX_PREDICTION_DRIFT = 60
KALMAN_MAX_VELOCITY_PX = 50

# Collision detection
# COLLISION_COOLDOWN_FRAMES: min frames between collision events
# COLLISION_MIN_APPROACH_SPEED: min relative speed (px/frame) magnitude to trigger
# COLLISION_MIN_CLOSING_SPEED: min approach speed along center-to-center axis;
#   positive = approaching. Rejects parallel fly-bys and tilt-induced overlap.
# COLLISION_MIN_OVERLAP_PX: min overlap depth in pixels; rejects edge-grazing
# COLLISION_MAX_RADIUS_JUMP: max fractional radius change between frames (0-1);
#   rejects tilt artifacts where the detected circle inflates suddenly
# COLLISION_KALMAN_CONFIRM: when True, require velocity direction change vs Kalman
#   prediction (reflection) to confirm impact; reduces false positives
# COLLISION_VELOCITY_REVERSAL_MIN_SPEED: min speed (px/frame) to check direction
# COLLISION_VELOCITY_REVERSAL_COS: cos(angle) threshold; < 0.5 = > 60 deg deflection
COLLISION_COOLDOWN_FRAMES = 15
COLLISION_MIN_APPROACH_SPEED = 40.0
COLLISION_MIN_CLOSING_SPEED = 15.0
COLLISION_MIN_OVERLAP_PX = 3.0
COLLISION_MAX_RADIUS_JUMP = 0.5
COLLISION_KALMAN_CONFIRM = True
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

# Play mode (--play): countdown before each tracking round
# Seconds with no bey detected before restarting countdown
PLAY_NO_TRACK_TIMEOUT = 10.0
# Per-phase durations (seconds) for the Italian countdown
PLAY_COUNTDOWN_DURATIONS = {
    "tre": 1.0,
    "due": 1.0,
    "uno": 1.0,
    "prontii": 1.5,
    "lancio": 1.0,
}

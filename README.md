# Beyblade X Arena Tracker

This project is heavily vibe-coded and still in the testing phase.

Real-time tracking system for a Beyblade X arena: detects up to 2 beyblades via webcam, tracks position, radius and velocity, maintains identity across frames, detects collisions, and shows a debug overlay.

### Demo videos

This is the result (still in WIP):

![WIP demo](demo/demo_wip.gif)

[Video with OpenCV result](https://youtube.com/shorts/Kg8o_FC7uRk)

[More demos (YouTube playlist)](https://www.youtube.com/playlist?list=PLrNs8uiECbXatd9XZKk8QShOT4uTjQlCy)

## Requirements

- Python 3.10 or higher
- Webcam or video file; default live source uses `config.CAMERA_INDEX` (often `0`) with threaded capture when available

## Installation

1. Clone or enter the project folder:
   ```bash
   cd open_beybladex_ar_core
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running

Start the tracker:

```bash
python main.py
```

- **Exit**: press `q` in the webcam window to close.

CLI arguments:

| Flag | Description |
|------|-------------|
| `-v` | Video input |
| `-s` | Save output video |
| `-o` | Output video path (use with `-s`) |
| `-d` | Debug |
| `-e` | Trail/impact effects |
| `-w` | WebSocket for `open_beybladex_ar_web` SFX projection |
| `-p` | Play mode: Italian countdown before each tracking round (**requires** `-w`) |
| `-a` | Manually select arena ROI |
| `-c` | Configure arena: select red zone (center + edge) then rail mask polygon (8-12 pts along rail) |
| `-l` | Low-light preset: higher exposure/gain, aggressive CLAHE, relaxed detection thresholds |

If you have the sibling [open_beybladex_ar_web](https://github.com/Trafitto/open-beybladex-ar-web) repo next to this one, the [Makefile](Makefile) can start core + web together (expects `.venv` in both projects):

| Command | Description |
|---------|-------------|
| `make play` | HTTP server on the web app, open browser, then `main.py -w -p -e` (optional `LOW_LIGHT=1` for `-l`) |
| `make core` | `main.py -w -e` only |
| `make web` | Static server for the web app only |

**Rail mask** (optional, via `-c`): A polygon defines the **play area**; saturation is zeroed in the green rail outside it so detection ignores the rim. Use **8–12** clicks along the inner rail edge, then `[c]`. Points persist in `RAIL_MASK_POINTS_FILE` (see `RAIL_MASK_*` in `config.py`).

During execution you will see:
- Colored circles around detected beyblades (green and blue)
- Highlighted center and velocity arrow
- Label with ID and scalar speed
- **IMPACT!** text when the collision pipeline confirms an impact (overlap plus filters such as closing speed and minimum overlap; see `CollisionDetector` in [`physics.py`](physics.py)), not on every brief geometric touch.

## Processing overview

1. **Input**: live camera (`config.CAMERA_INDEX` or threaded stream) or video file (`-v`).
2. **Preprocess** (optional): HSV + CLAHE on V and optional saturation scaling ([`preprocess.py`](preprocess.py)), controlled by `HSV_*` in [`config.py`](config.py).
3. **Arena geometry**: ROI, red/green priority zones, rail polygon, dome glare wedge—configured once or via `-c` / `-a`.
4. **Detection channel**: inverted grayscale and/or saturation (and related boosts) per config; rail and dome masks remove static rail and specular regions.
5. **Candidates**: **`DETECTION_METHOD == "contour"`** (default)—global threshold plus optional adaptive threshold, morphology, contour filtering; or **`"hough"`**—`cv2.HoughCircles` with tuned radii.
6. **Tracks**: match detections to existing beys by distance + **median chip hue** (saturated pixels), Kalman correction/prediction, optional circular-orbit prediction, motion-based registration for new beys.
7. **Physics**: `CollisionDetector` requires overlap, then applies debouncing, closing speed, overlap depth, relative speed, optional Kalman deflection check, and radius-jump checks before emitting an event; wall hits use the learned rim circle.
8. **Output**: on-screen overlay (`-d`), effects (`-e`), optional recording (`-s` / `-o`), WebSocket JSON (`-w`), optional play-mode countdown payload (`-p`).

## Configuration

All tunable parameters are in `config.py`. Use `-d` at runtime to see live detection counts and tracking state.

Run tests: `pytest tests/ -v`

### Important variables and how they affect tracking

| Variable | Effect of increasing | Effect of decreasing |
|----------|----------------------|------------------------|
| **DETECTION_METHOD** | Use `"hough"` for Hough circle detection | Use `"contour"` for threshold + contour pipeline (default); tune `CONTOUR_*`, `ADAPTIVE_THRESH_*` |
| **HOUGH_DETECTION_CHANNEL** | `"saturation"` = HSV S (better for white floor vs colored chips); `"grayscale"` = luminance | |
| **HOUGH_SAT_SCALE** | Boost colored chips vs white floor (1.2-1.5); higher = stronger contrast | Lower = raw saturation |
| **HOUGH_SAT_FLOOR** | Clip values below this to 0; suppress near-white noise (0 = disabled) | |
| **HOUGH_SAT_CLAHE_ENABLED** | Apply CLAHE to S channel for uneven arena lighting | False = no CLAHE on S |
| **HOUGH_PARAM2** | Fewer circle detections, cleaner but may miss blurry beys | More detections, noisier (false circles from edges, glare) |
| **HOUGH_MIN_RADIUS / HOUGH_MAX_RADIUS** | Only larger circles detected | Only smaller circles detected; adjust to match bey size in pixels |
| **HOUGH_MIN_DIST** | Fewer duplicate circles for same bey; may miss when 2 beys are close | More circles, risk of double-tracking one bey |
| **ARENA_ROI** radius (3rd value) | Larger search area; includes stadium perimeter | Smaller area; focuses on white floor only |
| **ARENA_ROI_HIGH** | Red circle, high-priority zone (center) | Set ARENA_ROI_LOW = None for red only |
| **ARENA_ROI_LOW** | Green circle; set to None to disable | |
| **PREFER_HIGH_PRIORITY** | When full, replace edge bey with unmatched center candidate | False = never replace |
| **REJECT_HUE_RANGES** | Add more hue ranges to exclude (e.g. `[(35, 95)]` for green rail) | Fewer exclusions; set `[]` to disable (needed for green beys) |
| **REJECT_NEAR_RIM_FRACTION** | Reject circles in outer X of arena; 0.10 = outer 10% (green rail zone) | 0 = disabled; lower = allow beys nearer rim |
| **RAIL_MASK_ENABLED** | Zero S in green rail region before detection (stadium is static) | False = no rail mask |
| **RAIL_MASK_POINTS_FILE** | JSON file for polygon points; load when exists, recreate with `-c` | `output/rail_mask_points.json` |
| **POLYGON_EDGE_MARGIN** | Reject circles within N px of polygon edge (rail reflections); 0 = disabled | 18 |
| **DOME_GLARE_ENABLED** | Zero S in specular spots (plastic dome reflections) before detection | True |
| **DOME_GLARE_V_MIN** | V above this = potential glare; lower = catch more | 200 |
| **DOME_GLARE_S_MAX** | S below this in bright region = specular; higher = catch more | 55 |
| **DOME_EXCLUDE_WEDGE_ENABLED** | Exclude angular wedge where "Beyblade X" text is (0=top, 90=right, 180=bottom) | False |
| **DEBUG_HIDE_RED_CIRCLE_WHEN_POLYGON** | Cleaner overlay: hide red circle when polygon ROI is used | True |
| **ZERO_VELOCITY_CLEAR_FRAMES** | Drop bey if speed < threshold for N frames (likely wrong object) | 0 = disabled |
| **COLOR_SAT_MIN** | Stricter: only vivid chips accepted | More permissive: paler chips accepted, risk of white glare |
| **COLOR_CENTER_MIN_FILL** | Stricter: more of center must be coloured | More permissive for small chips or motion blur |
| **MATCH_MAX_DISTANCE** | Tracks faster motion; may wrong-match when beys cross | More stable identity; may lose track when bey moves fast |
| **MATCH_IDENTITY_WEIGHT** | Color matters more; less likely to swap bey1/bey2 when they cross | Position dominates; may swap identities when beys cross |
| **IDENTITY_HUE_MAX_DRIFT** | Stricter hue match (set 0 to disable reject) | No hue-based rejection; any hue accepted |
| **IDENTITY_BOOTSTRAP_FRAMES** | More samples to lock identity; more robust to spin/incline | Identity locked from first frame only; may be noisy |
| **MAX_RECOVERY_FRAMES** | Keeps track longer when detection fails briefly | Drops track sooner; useful if tracker sticks to wrong object |
| **KALMAN_MAX_PREDICTION_DRIFT** | Prediction can travel further when detection misses | Prediction stays closer to last position; better for slow beys |
| **KALMAN_MAX_VELOCITY_PX** | Allow higher extrapolated velocity when lost | Reject prediction and hold; rely on detection to re-acquire (set 0 to disable) |
| **KALMAN_RIM_CLAMP_FRAC** | When bey is in outer band (dist >= this * radius), remove outward velocity | 0.92 = clamp in outer 8%; 0 = disabled |
| **CIRCULAR_PREDICTION_ENABLED** | Predict along circular arc (orbit) when detectable | Use linear (constant velocity) only |
| **CIRCULAR_HISTORY_LEN** | More positions for circle fit; smoother orbit estimate | Fewer; faster to adapt, less stable fit |
| **KALMAN_LOSS_VELOCITY_DECAY** | Prediction keeps velocity when detection is lost | Prediction slows quickly; better when detections are reliable |
| **BOOTSTRAP_MIN_SPEED** | Bey must move faster to count as "active" | Easier to trigger stuck-clear and rescan |
| **MAX_STUCK_FRAMES** | Waits longer before clearing and rescanning | Clears and rescans sooner when both beys stop |

### Quick fixes

- **Bey1 and bey2 swap when they cross**: Raise `MATCH_IDENTITY_WEIGHT` (e.g. `8`–`12`), set `IDENTITY_HUE_MAX_DRIFT` (e.g. `35`) to reject bad matches, increase `IDENTITY_BOOTSTRAP_FRAMES` (e.g. `15`) for a stable identity.
- **Tracker runs away when bey is briefly lost**: Set `KALMAN_MAX_VELOCITY_PX` (e.g. `60`–`100`) to reject extreme predictions and hold position until detection re-acquires the bey.
- **Tracking stadium rail instead of beys**: Enable `RAIL_MASK_ENABLED = True` (zeros saturation in green rail), set `REJECT_NEAR_RIM_FRACTION = 0.10`, or add `REJECT_HUE_RANGES = [(35, 95)]` (only if not using green beys).
- **Beys blur and disappear**: Lower `HOUGH_PARAM2` (e.g. `12`–`14`), raise `MATCH_MAX_DISTANCE` and `KALMAN_MAX_PREDICTION_DRIFT`.
- **Tracker locks onto wrong objects**: Enable `RAIL_MASK_ENABLED`, set `ZERO_VELOCITY_CLEAR_FRAMES = 45` (drop stationary tracks), shrink `ARENA_ROI`, lower `MAX_RECOVERY_FRAMES`.
- **Using green Beyblades**: Set `REJECT_HUE_RANGES = []` so green chips are not rejected.

## Web SFX Projection

Sibling project [open_beybladex_ar_web](https://github.com/Trafitto/open-beybladex-ar-web) provides a browser-based SFX output for projection. Run the core with `-w`/`--web` to broadcast tracking data via WebSocket. See the web project's README for setup.

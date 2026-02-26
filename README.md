# Beyblade X Arena Tracker

This project is heavily vibe-coded and still in the testing phase.

Real-time tracking system for a Beyblade X arena: detects up to 2 beyblades via webcam, tracks position, radius and velocity, maintains identity across frames, detects collisions, and shows a debug overlay.

### Demo videos

The first demo shows the tracking output with the debug overlay enabled. The second demo shows the same tracker with simple visual effects applied for testing.

![Demo 1](demo/demo1.gif)
![Demo 2](demo/demo2.gif)

The core tracker broadcasts bey position, velocity and collision events to the sibling project `open_beybladex_ar_web` via WebSocket (`-w`). That project renders the SFX overlay and projection in the browser. The video below shows the output captured from the web client.
![Demo 2](demo/test_video_output.gif)

## Requirements

- Python 3.10 or higher
- Webcam (the program uses `cv2.VideoCapture(0)`)

## Installation

1. Clone or enter the project folder:
   ```bash
   cd open_beybladex_ar_core
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Linux/macOS
   # .venv\Scripts\activate   # Windows
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

CLI arguments: `-v` video input, `-s` save output, `-d` debug, `-e` trail/impact effects, `-w` WebSocket for `open_beybladex_ar_web` SFX projection.

During execution you will see:
- Colored circles around detected beyblades (green and blue)
- Highlighted center and velocity arrow
- Label with ID and scalar speed
- **IMPACT!** text at center when the two beyblades touch (center distance < sum of radii)

## Pipeline

```
[Webcam / Video] --> [Frame] --> [Arena Mask (HSV)]
                                        |
                                        v
                              [Hough Circles Detection]
                                        |
                                        v
                              [Identity Assignment]
                              - nearest-neighbour + prediction
                              - reference histograms (re-id)
                              - recovery if stuck at edge
                                        |
                                        v
                              [BeyState: position, velocity, radius]
                                        |
                +-----------------------+-----------------------+
                v                       v                       v
        [Collision Detect]      [Overlay / Effects]      [WebSocket]
        (distance < r1+r2)     (trail, impact flash)    (open_beybladex_ar_web)
```

1. **Input**: webcam (`cv2.VideoCapture(0)`) or video file (`-v path`).
2. **Arena mask**: HSV mask that excludes white floor and green arena border so Hough detects only beyblades.
3. **Hough circles**: circle detection with radii in range `HOUGH_MIN_RADIUS`-`HOUGH_MAX_RADIUS`, scaled with frame size.
4. **Identity assignment**: each circle is assigned to the nearest bey (or predicted position); reference HSV histograms bootstrapped from the first frame and updated on the fly stabilize id0/id1; recovery within `ASSIGN_RECOVERY_FRACTION` if a bey has no circle.
5. **BeyState**: position (smoothed), velocity, scaled radius.
6. **Collision**: distance between centers < r1 + r2.
7. **Output**: overlay circles/arrows, trail and impact flash (`-e`), video save (`-s`), WebSocket for web SFX (`-w`).

## Project Structure

| File / Dir      | Role                                                                 |
|-----------------|----------------------------------------------------------------------|
| `main.py`       | Entry point: argparse, main loop orchestration                       |
| `arena.py`      | Arena ROI setup (manual, config, auto-detect)                        |
| `preprocess.py` | Frame preprocessing (HSV, CLAHE)                                     |
| `roi.py`        | Interactive arena ROI selection UI                                  |
| `overlay/`      | Drawing: debug overlay, effects (trail, impact), main overlay       |
| `web.py`        | WebSocket server, `build_tracking_data`, push to clients            |
| `tracker.py`    | `BeyTracker`: Hough circles, identity, Kalman, bey state            |
| `physics.py`    | Velocity, collision and wall detection                              |
| `utils.py`      | Position smoothing, Euclidean distance                               |
| `config.py`     | Tunable parameters (camera, Hough, smoothing, colors, debug)         |
| `tests/`        | Unit tests for preprocess, physics, web, utils                      |

Run tests: `pytest tests/ -v`

## Configuration

All tunable parameters are in `config.py`. Use `-d` at runtime to see live detection counts and tracking state.

### Important variables and how they affect tracking

| Variable | Effect of increasing | Effect of decreasing |
|----------|----------------------|------------------------|
| **HOUGH_PARAM2** | Fewer circle detections, cleaner but may miss blurry beys | More detections, noisier (false circles from edges, glare) |
| **HOUGH_MIN_RADIUS / HOUGH_MAX_RADIUS** | Only larger circles detected | Only smaller circles detected; adjust to match bey size in pixels |
| **HOUGH_MIN_DIST** | Fewer duplicate circles for same bey; may miss when 2 beys are close | More circles, risk of double-tracking one bey |
| **ARENA_ROI** radius (3rd value) | Larger search area; includes stadium perimeter | Smaller area; focuses on white floor only |
| **ARENA_ROI_HIGH / LOW** | Dual ROI: high (red, center) checked first; if 2 beys there, done | Set HIGH = None to use single ARENA_ROI |
| **PREFER_HIGH_PRIORITY** | When full, replace edge bey with unmatched center candidate | False = never replace |
| **REJECT_HUE_RANGES** | Add more hue ranges to exclude (e.g. `[(98, 142), (0, 5)]`) | Fewer exclusions; set `[]` to disable (needed for green beys) |
| **COLOR_SAT_MIN** | Stricter: only vivid chips accepted | More permissive: paler chips accepted, risk of white glare |
| **COLOR_CENTER_MIN_FILL** | Stricter: more of center must be coloured | More permissive for small chips or motion blur |
| **MATCH_MAX_DISTANCE** | Tracks faster motion; may wrong-match when beys cross | More stable identity; may lose track when bey moves fast |
| **MATCH_IDENTITY_WEIGHT** | Color matters more; less likely to swap bey1/bey2 when they cross | Position dominates; may swap identities when beys cross |
| **IDENTITY_HUE_MAX_DRIFT** | Stricter hue match (set 0 to disable reject) | No hue-based rejection; any hue accepted |
| **IDENTITY_BOOTSTRAP_FRAMES** | More samples to lock identity; more robust to spin/incline | Identity locked from first frame only; may be noisy |
| **MAX_RECOVERY_FRAMES** | Keeps track longer when detection fails briefly | Drops track sooner; useful if tracker sticks to wrong object |
| **KALMAN_MAX_PREDICTION_DRIFT** | Prediction can travel further when Hough misses | Prediction stays closer to last position; better for slow beys |
| **KALMAN_MAX_VELOCITY_PX** | Allow higher extrapolated velocity when lost | Reject prediction and hold; rely on Hough to re-acquire (set 0 to disable) |
| **KALMAN_RIM_CLAMP_FRAC** | When bey is in outer band (dist >= this * radius), remove outward velocity | 0.92 = clamp in outer 8%; 0 = disabled |
| **CIRCULAR_PREDICTION_ENABLED** | Predict along circular arc (orbit) when detectable | Use linear (constant velocity) only |
| **CIRCULAR_HISTORY_LEN** | More positions for circle fit; smoother orbit estimate | Fewer; faster to adapt, less stable fit |
| **KALMAN_LOSS_VELOCITY_DECAY** | Prediction keeps velocity when detection is lost | Prediction slows quickly; better when detections are reliable |
| **BOOTSTRAP_MIN_SPEED** | Bey must move faster to count as "active" | Easier to trigger stuck-clear and rescan |
| **MAX_STUCK_FRAMES** | Waits longer before clearing and rescanning | Clears and rescans sooner when both beys stop |

### Quick fixes

- **Bey1 and bey2 swap when they cross**: Raise `MATCH_IDENTITY_WEIGHT` (e.g. `8`–`12`), set `IDENTITY_HUE_MAX_DRIFT` (e.g. `35`) to reject bad matches, increase `IDENTITY_BOOTSTRAP_FRAMES` (e.g. `15`) for a stable identity.
- **Tracker runs away when bey is briefly lost**: Set `KALMAN_MAX_VELOCITY_PX` (e.g. `60`–`100`) to reject extreme predictions and hold position until Hough re-acquires the circle.
- **Tracking stadium rail instead of beys**: Lower `ARENA_ROI` radius (e.g. `0.35`) or add `REJECT_HUE_RANGES = [(98, 142)]` for the green X-rail.
- **Beys blur and disappear**: Lower `HOUGH_PARAM2` (e.g. `12`–`14`), raise `MATCH_MAX_DISTANCE` and `KALMAN_MAX_PREDICTION_DRIFT`.
- **Tracker locks onto wrong objects**: Shrink `ARENA_ROI`, tune `REJECT_HUE_RANGES`, lower `MAX_RECOVERY_FRAMES` so it resets sooner.
- **Using green Beyblades**: Set `REJECT_HUE_RANGES = []` so green chips are not rejected.

## Web SFX Projection

Sibling project [open_beybladex_ar_web](https://github.com/Trafitto/open-beybladex-ar-web) provides a browser-based SFX output for projection. Run the core with `-w`/`--web` to broadcast tracking data via WebSocket. See the web project's README for setup.

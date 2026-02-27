"""
One-shot script: capture first frame, build rail mask, save and open.
Run: python run_rail_mask_snapshot.py
"""
import os
import subprocess
import sys

import cv2

# Ensure we save the mask
import config
config.RAIL_MASK_SAVE_PATH = "output/rail_mask.png"

from arena import setup_arena_roi
from preprocess import preprocess_frame_hsv_from_config
from roi import load_rail_mask_points
from tracker import BeyTracker


def main() -> None:
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        print("Cannot open camera")
        sys.exit(1)

    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Cannot read frame")
        sys.exit(1)

    frame = preprocess_frame_hsv_from_config(frame, config)
    tracker = BeyTracker()
    setup_arena_roi(tracker, frame, manual=False)

    points_file = getattr(config, "RAIL_MASK_POINTS_FILE", "output/rail_mask_points.json")
    points = load_rail_mask_points(points_file, frame_shape=frame.shape)
    if points:
        ok = tracker.set_rail_mask_from_polygon(frame.shape, points, save_path=config.RAIL_MASK_SAVE_PATH)
    else:
        ok = tracker.build_rail_mask(frame)

    if ok:
        path = config.RAIL_MASK_SAVE_PATH
        if path and os.path.isfile(path):
            print(f"Opening {path}")
            if sys.platform == "linux":
                subprocess.run(["xdg-open", path], check=False)
            elif sys.platform == "darwin":
                subprocess.run(["open", path], check=False)
    else:
        print("Rail mask build failed")


if __name__ == "__main__":
    main()

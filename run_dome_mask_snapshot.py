"""
One-shot script: capture first frame, build dome mask (glare + wedge), save and open.
Run: python run_dome_mask_snapshot.py
"""
import os
import subprocess
import sys

import cv2
import numpy as np

import config

from arena import setup_arena_roi
from preprocess import preprocess_frame_hsv_from_config
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

    mask = tracker.get_dome_mask(frame)
    save_path = getattr(config, "DOME_MASK_SAVE_PATH", "output/dome_mask.png")

    if mask is None:
        print("Dome mask unavailable (set HOUGH_DETECTION_CHANNEL=saturation in config)")
        sys.exit(1)

    if np.sum(mask > 0) == 0:
        print("No pixels excluded. Try lowering DOME_GLARE_V_MIN or raising DOME_GLARE_S_MAX in config.")

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    if cv2.imwrite(save_path, mask):
        print(f"Dome mask saved to {save_path}")

    overlay_path = save_path.replace(".png", "_overlay.png")
    overlay = frame.copy()
    overlay[mask > 0] = (0, 128, 255)
    overlay_blend = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)
    if cv2.imwrite(overlay_path, overlay_blend):
        print(f"Dome overlay saved to {overlay_path}")

    if sys.platform == "linux":
        subprocess.run(["xdg-open", save_path], check=False)
    elif sys.platform == "darwin":
        subprocess.run(["open", save_path], check=False)


if __name__ == "__main__":
    main()

"""
Arena ROI setup: manual selection, config-based, or auto-detection.
"""
import os

import cv2

import config
from roi import select_arena_roi


def setup_arena_roi(
    tracker,
    first_frame,
    *,
    manual: bool = False,
    cfg=None,
) -> None:
    """
    Configure arena ROI on the tracker from manual selection, config, or auto-detection.

    Uses cfg if provided, otherwise config module.
    """
    cfg = cfg or config
    if manual:
        roi = select_arena_roi(first_frame)
        if roi:
            if len(roi) == 4:
                tracker.set_arena_roi_dual(*roi)
                print(f"Arena ROI high: center=({roi[0]},{roi[1]}) r={roi[2]}")
                print(f"Arena ROI low: center=({roi[0]},{roi[1]}) r={roi[3]}")
            else:
                tracker.set_arena_roi(*roi)
                print(f"Arena ROI selected: center=({roi[0]},{roi[1]}) r={roi[2]}")
        else:
            print("No ROI selected -- tracking entire frame")
        return

    roi_high_cfg = getattr(cfg, "ARENA_ROI_HIGH", None)
    roi_low_cfg = getattr(cfg, "ARENA_ROI_LOW", None)
    arena_roi_cfg = getattr(cfg, "ARENA_ROI", None)
    ref_dir = getattr(cfg, "ARENA_REFERENCE_DIR", None)

    fh, fw = first_frame.shape[:2]
    min_dim = min(fh, fw)

    if roi_high_cfg is not None and roi_low_cfg is not None:
        cx = int(roi_high_cfg[0] * fw)
        cy = int(roi_high_cfg[1] * fh)
        r_high = int(roi_high_cfg[2] * min_dim)
        r_low = int(roi_low_cfg[2] * min_dim)
        tracker.set_arena_roi_dual(cx, cy, r_high, r_low)
        print(
            f"Arena ROI dual: high r={r_high} (red) low r={r_low} "
            f"center=({cx},{cy})"
        )
        return

    if roi_high_cfg is not None and roi_low_cfg is None:
        cx = int(roi_high_cfg[0] * fw)
        cy = int(roi_high_cfg[1] * fh)
        r = int(roi_high_cfg[2] * min_dim)
        tracker.set_arena_roi_high_only(cx, cy, r)
        print(f"Arena ROI high only (red): center=({cx},{cy}) r={r}")
        return

    if arena_roi_cfg is not None:
        roi_cx = int(arena_roi_cfg[0] * fw)
        roi_cy = int(arena_roi_cfg[1] * fh)
        roi_r = int(arena_roi_cfg[2] * min_dim)
        tracker.set_arena_roi(roi_cx, roi_cy, roi_r)
        print(f"Arena ROI from config: center=({roi_cx},{roi_cy}) r={roi_r}")
        return

    if ref_dir and os.path.isdir(ref_dir):
        matched = _try_reference_detection(tracker, ref_dir, first_frame)
        if matched:
            return
        if tracker.auto_detect_roi_hough(first_frame):
            _print_roi_detected(tracker, "Hough")
        else:
            print("No reference matched and Hough failed -- tracking entire frame")
        return

    if tracker.auto_detect_roi_hough(first_frame):
        _print_roi_detected(tracker, "Hough")
    else:
        print("No ROI detected -- tracking entire frame")


def _try_reference_detection(tracker, ref_dir: str, first_frame) -> bool:
    """Try to detect ROI from reference images. Returns True if matched."""
    for fname in sorted(os.listdir(ref_dir)):
        fpath = os.path.join(ref_dir, fname)
        ref_img = cv2.imread(fpath)
        if ref_img is None:
            continue
        if tracker.auto_detect_roi_from_reference(ref_img, first_frame):
            roi = tracker.get_arena_roi()
            print(
                f"Arena ROI auto-detected via {fname}: "
                f"center=({roi[0]},{roi[1]}) r={roi[2]}"
            )
            return True
    if tracker.auto_detect_roi_hough(first_frame):
        _print_roi_detected(tracker, "Hough")
        return True
    return False


def _print_roi_detected(tracker, method: str) -> None:
    roi = tracker.get_arena_roi()
    if roi:
        print(f"Arena ROI auto-detected via {method}: center=({roi[0]},{roi[1]}) r={roi[2]}")

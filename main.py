"""
Main loop: webcam capture, tracking, physics, debug overlay.
Exit with 'q'.
Use -v/--video <path> to use a video file instead of the webcam.
Use -s/--save to write the overlay output to output/videos/.
Use -d/--debug to show tuning info overlay.
Use -e/--effect to enable trail SFX under each bey.
Use -w/--web to broadcast tracking data via WebSocket for open_beybladex_ar_web SFX projection.
"""
import argparse
import os
import time
from collections import deque
from datetime import datetime

import cv2

import config
from arena import setup_arena_roi
from video_stream import WebcamVideoStream, is_live_source
from overlay.debug import draw_debug_overlay_from_config
from roi import (
    load_rail_mask_points,
    load_red_zone,
    save_rail_mask_points,
    save_red_zone,
    select_rail_mask_points,
    select_red_zone,
)
from overlay.effects import (
    draw_impact_effect_from_config,
    draw_impact_label_from_config,
    draw_trail_effect_from_config,
    draw_wall_hit_from_config,
)
from overlay.overlay import draw_overlay_from_config
from physics import CollisionDetector, detect_wall_collisions
from preprocess import preprocess_frame_hsv_from_config
from tracker import BeyTracker
from utils import get_bey_label
from web import build_tracking_data, push_tracking_web, run_websocket_server


def _live_preview(cap, *, proc_w: int = 0, proc_h: int = 0) -> "np.ndarray | None":
    """Show a live camera feed until the user presses SPACE to freeze a frame.

    Returns the frozen (and optionally resized) frame, or None if the user
    pressed 'q' to abort.
    """
    import numpy as np  # local to avoid top-level import cost when unused

    win = "Live Preview - press SPACE to freeze, Q to abort"
    print("Live preview: press SPACE to freeze the frame for calibration, Q to abort")
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        if proc_w > 0 and proc_h > 0:
            fh, fw = frame.shape[:2]
            if fw != proc_w or fh != proc_h:
                frame = cv2.resize(frame, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
        cv2.imshow(win, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            cv2.destroyWindow(win)
            return frame
        if key == ord("q"):
            cv2.destroyWindow(win)
            return None


def _run_main_loop(
    cap,
    tracker: BeyTracker,
    collision_detector: CollisionDetector,
    *,
    args,
    is_live: bool,
) -> None:
    video_fps = cap.get(cv2.CAP_PROP_FPS) if not is_live else None
    # For video files: frame delay handled by cv2.waitKey(delay_ms)
    # For live sources: cv2.waitKey(1) (non-blocking, just poll for 'q')
    wait_ms = max(1, int(1000.0 / video_fps)) if video_fps else 1

    prev_time = time.time()
    out_writer = None
    out_path = None
    trail_history: dict[int, deque] = {}
    impact_effect_remaining = 0
    impact_center: tuple[float, float] = (0.0, 0.0)
    impact_force: float = 0.0
    wall_hit_remaining: dict[int, int] = {}
    frame_index = 0
    ws_setter = None
    if args.web:
        ws_host = getattr(config, "WEB_WS_HOST", "127.0.0.1")
        ws_port = getattr(config, "WEB_WS_PORT", 8765)
        ws_setter = run_websocket_server(ws_host, ws_port)
        time.sleep(0.3)

    proc_w = getattr(config, "PROCESS_WIDTH", 0)
    proc_h = getattr(config, "PROCESS_HEIGHT", 0)

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        if proc_w > 0 and proc_h > 0:
            fh, fw = frame.shape[:2]
            if fw != proc_w or fh != proc_h:
                frame = cv2.resize(frame, (proc_w, proc_h), interpolation=cv2.INTER_AREA)

        frame_index += 1
        now = time.time()
        dt = now - prev_time
        prev_time = now
        if dt <= 0:
            dt = 1.0 / (video_fps or config.TARGET_FPS)

        frame = preprocess_frame_hsv_from_config(frame, config)
        tracker.update(frame, dt)
        states = tracker.get_states()
        positions = [b.position for b in states]
        velocities = [b.velocity for b in states]
        collision_margin = max(getattr(config, "BEY_COLLISION_MARGIN_PX", 2), 1)
        radii = [b.radius + collision_margin for b in states]

        predicted_velocities = [
            getattr(b, "predicted_velocity", None) for b in states
        ]
        collision_event = collision_detector.update(
            positions,
            radii,
            velocities,
            frame_index,
            predicted_velocities=predicted_velocities,
        )
        collision = collision_event is not None

        if collision and config.DEBUG_PRINT:
            print(
                f"IMPACT #{collision_detector.event_count} "
                f"force={collision_event.impact_force:.0f} "
                f"speed={collision_event.relative_speed:.0f}"
            )

        if collision and collision_event is not None:
            impact_center = collision_event.position
            impact_force = collision_event.impact_force
            if args.effect:
                impact_effect_remaining = getattr(config, "IMPACT_FLASH_FRAMES", 8)

        rim_circle = tracker.get_rim_circle()
        wall_hits: list[int] = []
        if rim_circle is not None and len(positions) > 0:
            margin = getattr(config, "WALL_COLLISION_MARGIN", 5)
            wall_hits = detect_wall_collisions(positions, radii, rim_circle, margin)
            for idx in wall_hits:
                bid = states[idx].id
                if bid not in wall_hit_remaining or wall_hit_remaining[bid] <= 0:
                    wall_hit_remaining[bid] = getattr(config, "WALL_FLASH_FRAMES", 6)
                    if config.DEBUG_PRINT:
                        print(f"WALL HIT bey#{bid}")

        if args.effect:
            draw_trail_effect_from_config(frame, states, trail_history, config)
            if impact_effect_remaining > 0:
                draw_impact_effect_from_config(
                    frame, impact_center, impact_effect_remaining, impact_force, config
                )
                draw_impact_label_from_config(
                    frame, impact_center, impact_effect_remaining, impact_force, config
                )
                impact_effect_remaining -= 1
            if rim_circle is not None:
                for b in states:
                    if wall_hit_remaining.get(b.id, 0) > 0:
                        draw_wall_hit_from_config(
                            frame, b, rim_circle, wall_hit_remaining[b.id], config
                        )
                        wall_hit_remaining[b.id] -= 1

        if args.debug:
            draw_overlay_from_config(
                frame, states, collision,
                impact_center if collision else None,
                config,
            )
            draw_debug_overlay_from_config(
                frame, tracker, collision_detector, frame_index, config
            )

        if args.web and ws_setter:
            h, w = frame.shape[:2]
            sorted_states = sorted(states, key=lambda b: b.id)
            identities = [
                get_bey_label(b, i, getattr(config, "BEY_IDENTITIES", None))
                for i, b in enumerate(sorted_states)
            ]
            arena_center = tracker.get_arena_center_px()
            data = build_tracking_data(
                w, h, states, collision, impact_center, wall_hits,
                collision_detector.event_count,
                collision_event=collision_event,
                radius_scale=config.BEY_RADIUS_SCALE,
                identities=identities,
                mm_per_pixel=tracker.mm_per_pixel,
                arena_center_px=arena_center,
                arena_radius_px=tracker.get_arena_radius_px(),
                wall_hit_tolerance_mm=getattr(config, "WALL_HIT_TOLERANCE_MM", 15.0),
            )
            push_tracking_web(data, ws_setter)

        if args.save and out_writer is None:
            if getattr(args, "output", None):
                out_path = args.output
                os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            else:
                out_dir = "output/videos"
                os.makedirs(out_dir, exist_ok=True)
                out_name = f"arena_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
                out_path = os.path.join(out_dir, out_name)
            fps = video_fps or config.TARGET_FPS
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out_writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
            print("Saving to", out_path)
        if out_writer is not None:
            out_writer.write(frame)

        cv2.imshow("Beyblade X Arena", frame)

        if cv2.waitKey(wait_ms) & 0xFF == ord("q"):
            break

    total_collisions = collision_detector.event_count
    if total_collisions > 0:
        print(f"\nSession summary: {total_collisions} collision(s) detected")
        for i, evt in enumerate(collision_detector.events):
            print(
                f"  #{i+1} frame={evt.frame} "
                f"pos=({evt.position[0]:.0f},{evt.position[1]:.0f}) "
                f"rel_speed={evt.relative_speed:.0f} "
                f"force={evt.impact_force:.0f}"
            )
    else:
        print("\nSession summary: no collisions detected")

    cap.release()
    if out_writer is not None:
        out_writer.release()
        print("Saved", out_path)
    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Beyblade X Arena tracker")
    parser.add_argument("-v", "--video", metavar="PATH", help="Use video file instead of webcam")
    parser.add_argument("-s", "--save", action="store_true", help="Save video output to output/videos/")
    parser.add_argument("-o", "--output", metavar="PATH", help="Output video path (use with -s)")
    parser.add_argument("-d", "--debug", action="store_true", help="Show tracking overlay and tuning params")
    parser.add_argument("-e", "--effect", action="store_true", help="Enable trail SFX under each bey")
    parser.add_argument("-w", "--web", action="store_true", help="Broadcast tracking data via WebSocket")
    parser.add_argument("-a", "--arena", action="store_true", help="Manually select the arena ROI on the first frame")
    parser.add_argument(
        "-rm", "--rail-mask",
        action="store_true",
        help="Manually select rail mask: click points along the green rail (8-12 pts), then [c]"
    )
    parser.add_argument(
        "-rz", "--red-zone",
        action="store_true",
        help="Manually select red zone: click center, then edge of circle, then [c]"
    )
    args = parser.parse_args()

    proc_w = getattr(config, "PROCESS_WIDTH", 0)
    proc_h = getattr(config, "PROCESS_HEIGHT", 0)

    source = args.video if args.video else config.CAMERA_INDEX
    is_live = is_live_source(source)

    if is_live:
        try:
            cap = WebcamVideoStream(
                source,
                target_fps=config.TARGET_FPS,
                width=proc_w,
                height=proc_h,
            ).start()
        except RuntimeError as exc:
            print(exc)
            return
        print(f"Live source: {source} (threaded capture)")
    else:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print("Cannot open video:", source)
            return
        fps = cap.get(cv2.CAP_PROP_FPS) or config.TARGET_FPS
        print(f"Video file: {source} (frame-paced at {fps:.1f} fps)")

    tracker = BeyTracker()
    collision_detector = CollisionDetector()

    needs_calibration = args.red_zone or args.rail_mask
    if is_live and needs_calibration:
        first_frame = _live_preview(cap, proc_w=proc_w, proc_h=proc_h)
        if first_frame is None:
            print("Live preview aborted")
            cap.release()
            return
    else:
        ret, first_frame = cap.read()
        if not ret or first_frame is None:
            print("Cannot read first frame")
            cap.release()
            return
        if proc_w > 0 and proc_h > 0:
            fh, fw = first_frame.shape[:2]
            if fw != proc_w or fh != proc_h:
                first_frame = cv2.resize(first_frame, (proc_w, proc_h), interpolation=cv2.INTER_AREA)

    first_frame = preprocess_frame_hsv_from_config(first_frame, config)
    setup_arena_roi(tracker, first_frame, manual=args.arena)

    red_zone_file = getattr(config, "RED_ZONE_POINTS_FILE", "output/red_zone.json")
    if args.red_zone:
        red_zone = select_red_zone(first_frame)
        if red_zone:
            cx, cy, r = red_zone
            save_red_zone(red_zone_file, cx, cy, r, frame_shape=first_frame.shape)
            tracker.set_arena_roi_high_only(cx, cy, r)
            print(f"Red zone saved: center=({cx},{cy}) r={r}")
        else:
            print("Red zone selection cancelled")
    else:
        red_zone = load_red_zone(red_zone_file, frame_shape=first_frame.shape)
        if red_zone:
            cx, cy, r = red_zone
            tracker.set_arena_roi_high_only(cx, cy, r)
            print(f"Red zone loaded: center=({cx},{cy}) r={r}")

    points_file = getattr(config, "RAIL_MASK_POINTS_FILE", "output/rail_mask_points.json")
    if args.rail_mask:
        points = select_rail_mask_points(first_frame)
        if points:
            save_rail_mask_points(points_file, points, frame_shape=first_frame.shape)
            mask_path = getattr(config, "RAIL_MASK_SAVE_PATH", "") or "output/rail_mask.png"
            if tracker.set_rail_mask_from_polygon(first_frame.shape, points, save_path=mask_path):
                print("Rail mask set from polygon (tracking area = inside)")
        else:
            print("Rail mask selection cancelled or too few points")
    else:
        points = load_rail_mask_points(points_file, frame_shape=first_frame.shape)
        if points:
            tracker.set_rail_mask_from_polygon(first_frame.shape, points)
            print("Rail mask loaded from", points_file)
        elif getattr(config, "RAIL_MASK_ENABLED", False):
            if tracker.build_rail_mask(first_frame):
                print("Rail mask built from first frame (green rail region masked)")
            else:
                print("Rail mask build failed; run with -rm to define polygon")

    if not is_live:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    _run_main_loop(cap, tracker, collision_detector, args=args, is_live=is_live)


if __name__ == "__main__":
    main()

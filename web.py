"""
WebSocket server and tracking data serialization for web SFX projection.
"""
import asyncio
import json
import sys
import threading
from typing import Callable, Optional


def _normalize_force(raw_force: float, max_force: float = 5000.0) -> float:
    """Map raw impact_force to [0.0, 1.0] using soft clamping (tanh curve).

    max_force controls where the curve reaches ~0.76; forces beyond that
    asymptotically approach 1.0, preventing AR effects from clipping.
    """
    import math
    if raw_force <= 0:
        return 0.0
    return min(1.0, math.tanh(raw_force / max_force))


def build_tracking_data(
    frame_w: int,
    frame_h: int,
    states: list,
    collision: bool,
    impact_center: tuple[float, float],
    wall_hits: list[int] | None = None,
    collision_count: int = 0,
    collision_event=None,
    radius_scale: float = 1.4,
    blade_radius_px: float = 17.0,
    identities: Optional[list[str]] = None,
    mm_per_pixel: float = 0.0,
    arena_center_px: tuple[float, float] | None = None,
    arena_radius_px: float = 0.0,
    wall_hit_tolerance_mm: float = 15.0,
) -> dict:
    """
    Build JSON-serializable tracking payload for WebSocket clients.

    Pure function: no side effects, easy to unit test.
    Web expects id 0/1 for trail slots; use slot index as id.

    When *mm_per_pixel* > 0 the payload includes real-world units:
      speedMmS, vxMmS, vyMmS, kineticEnergyMm (mm, mm/s, mm^2/s^2)
    When *arena_center_px* and *arena_radius_px* are set, each bey
    gets a ``wallHit`` boolean (True when close to the stadium rim).
    """
    import math

    scale = mm_per_pixel
    has_arena = arena_center_px is not None and arena_radius_px > 0
    wall_tol_px = (wall_hit_tolerance_mm / scale) if scale > 0 else 0.0

    beys = []
    sorted_states = sorted(states, key=lambda b: b.id)
    id_list = identities or []
    wall_hit_idx_set = set(wall_hits) if wall_hits else set()

    for slot, b in enumerate(sorted_states):
        x, y = b.position[0], b.position[1]
        nx = x / frame_w if frame_w > 0 else 0
        ny = y / frame_h if frame_h > 0 else 0
        identity = id_list[slot] if slot < len(id_list) else ""

        speed_px = b.speed
        vx_px, vy_px = b.velocity
        ke_px = 0.5 * speed_px * speed_px

        entry: dict = {
            "id": slot,
            "identity": identity,
            "x": round(x, 2),
            "y": round(y, 2),
            "nx": round(nx, 4),
            "ny": round(ny, 4),
            "vx": round(vx_px, 2),
            "vy": round(vy_px, 2),
            "speed": round(speed_px, 2),
            "radius": round(blade_radius_px * radius_scale, 2),
            "kineticEnergy": round(ke_px, 2),
        }

        if scale > 0:
            speed_mm = speed_px * scale
            entry["speedMmS"] = round(speed_mm, 2)
            entry["vxMmS"] = round(vx_px * scale, 2)
            entry["vyMmS"] = round(vy_px * scale, 2)
            entry["kineticEnergyMm"] = round(0.5 * speed_mm * speed_mm, 2)

        wall_hit = slot in wall_hit_idx_set
        if has_arena and not wall_hit:
            acx, acy = arena_center_px
            dx = x - acx
            dy = y - acy
            dist = math.sqrt(dx * dx + dy * dy)
            bey_r_px = blade_radius_px
            wall_hit = (dist + bey_r_px) >= (arena_radius_px - wall_tol_px)
        entry["wallHit"] = wall_hit

        beys.append(entry)

    icx, icy = impact_center[0], impact_center[1]

    impact_data: dict = {
        "x": round(icx, 2),
        "y": round(icy, 2),
        "nx": round(icx / frame_w, 4) if frame_w > 0 else 0,
        "ny": round(icy / frame_h, 4) if frame_h > 0 else 0,
    }

    if collision_event is not None:
        rel_speed = collision_event.relative_speed
        raw_force = collision_event.impact_force
        impact_data["relativeSpeed"] = round(rel_speed, 2)
        impact_data["impactForceRaw"] = round(raw_force, 2)
        impact_data["impactForce"] = round(
            _normalize_force(raw_force), 4
        )
        if scale > 0:
            impact_data["relativeSpeedMmS"] = round(rel_speed * scale, 2)

    return {
        "frameWidth": frame_w,
        "frameHeight": frame_h,
        "mmPerPixel": round(scale, 6) if scale > 0 else None,
        "beys": beys,
        "collision": collision,
        "impactCenter": impact_data,
        "wallHits": wall_hits or [],
        "collisionCount": collision_count,
    }


def run_websocket_server(
    host: str,
    port: int,
    on_set_latest: Callable[[Callable[[str], None]], None] | None = None,
) -> Callable[[str], None]:
    """
    Start WebSocket server in a background thread.

    Returns a setter function to push JSON strings to all connected clients.
    Stores the setter on push_tracking_web._default_setter for use when no
    setter is passed. If on_set_latest is provided, it is called with the
    setter (for testing).
    """
    try:
        from websockets.asyncio.server import serve
    except ImportError:
        try:
            from websockets.server import serve
        except ImportError:
            from websockets import serve

    latest: dict = {"json": None}
    clients: set = set()

    async def register(ws):
        clients.add(ws)
        try:
            await ws.wait_closed()
        finally:
            clients.discard(ws)

    async def broadcast_loop():
        last = None
        while True:
            await asyncio.sleep(1 / 60)
            payload = latest.get("json")
            if payload is not None and payload != last:
                last = payload
                dead = []
                for c in clients:
                    try:
                        await c.send(payload)
                    except Exception:
                        dead.append(c)
                for c in dead:
                    clients.discard(c)

    async def main():
        async with serve(register, host, port):
            asyncio.create_task(broadcast_loop())
            await asyncio.Future()

    def set_latest(s: str):
        latest["json"] = s

    push_tracking_web._default_setter = set_latest
    if on_set_latest:
        on_set_latest(set_latest)

    def run():
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(main())
        except Exception as e:
            print(f"WebSocket server error: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    print(f"WebSocket server on ws://{host}:{port} (start core first, then open web page)")
    return set_latest


def push_tracking_web(data: dict, setter: Callable[[str], None] | None = None) -> None:
    """
    Push tracking data as JSON to WebSocket clients.

    If setter is None, uses the default global from run_websocket_server.
    Pass setter explicitly for testability.
    """
    payload = json.dumps(data, separators=(",", ":"))
    if setter is not None:
        setter(payload)
    else:
        _default_setter = getattr(push_tracking_web, "_default_setter", None)
        if _default_setter is not None:
            _default_setter(payload)

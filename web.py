"""
WebSocket server and tracking data serialization for web SFX projection.
"""
import asyncio
import json
import sys
import threading
from typing import Callable


def build_tracking_data(
    frame_w: int,
    frame_h: int,
    states: list,
    collision: bool,
    impact_center: tuple[float, float],
    wall_hits: list[int] | None = None,
    collision_count: int = 0,
    radius_scale: float = 1.4,
) -> dict:
    """
    Build JSON-serializable tracking payload for WebSocket clients.

    Pure function: no side effects, easy to unit test.
    """
    beys = []
    for b in states:
        x, y = b.position[0], b.position[1]
        nx = x / frame_w if frame_w > 0 else 0
        ny = y / frame_h if frame_h > 0 else 0
        beys.append({
            "id": b.id,
            "x": round(x, 2),
            "y": round(y, 2),
            "nx": round(nx, 4),
            "ny": round(ny, 4),
            "vx": round(b.velocity[0], 2),
            "vy": round(b.velocity[1], 2),
            "speed": round(b.speed, 2),
            "radius": round(b.radius * radius_scale, 2),
        })
    icx, icy = impact_center[0], impact_center[1]
    return {
        "frameWidth": frame_w,
        "frameHeight": frame_h,
        "beys": beys,
        "collision": collision,
        "impactCenter": {
            "x": round(icx, 2),
            "y": round(icy, 2),
            "nx": round(icx / frame_w, 4) if frame_w > 0 else 0,
            "ny": round(icy / frame_h, 4) if frame_h > 0 else 0,
        },
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

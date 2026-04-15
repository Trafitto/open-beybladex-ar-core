"""
Play-mode state machine for the Beyblade X countdown sequence.

Manages the "Tre, due, uno, prontii... LANCIO!" countdown that is
displayed on the web projection before each round of tracking.
"""
import enum
import time
from typing import Optional

import config


class PlayState(enum.Enum):
    DISABLED = "disabled"
    WAITING_CONNECTION = "waiting_connection"
    COUNTDOWN = "countdown"
    TRACKING = "tracking"


class CountdownPhase(enum.Enum):
    TRE = "tre"
    DUE = "due"
    UNO = "uno"
    PRONTII = "prontii"
    LANCIO = "lancio"


_PHASE_ORDER = [
    CountdownPhase.TRE,
    CountdownPhase.DUE,
    CountdownPhase.UNO,
    CountdownPhase.PRONTII,
    CountdownPhase.LANCIO,
]

_PHASE_DISPLAY_TEXT = {
    CountdownPhase.TRE: "TRE",
    CountdownPhase.DUE: "DUE",
    CountdownPhase.UNO: "UNO",
    CountdownPhase.PRONTII: "PRONTII...",
    CountdownPhase.LANCIO: "LANCIO!",
}


def _phase_duration(phase: CountdownPhase) -> float:
    """Return the duration in seconds for a countdown phase."""
    durations = getattr(config, "PLAY_COUNTDOWN_DURATIONS", {})
    return durations.get(phase.value, 1.0)


class PlayModeController:
    """State machine that drives the play-mode countdown loop.

    Lifecycle (when enabled):
        WAITING_CONNECTION  --[ws client connects]-->  COUNTDOWN
        COUNTDOWN           --[LANCIO ends]--------->  TRACKING
        TRACKING            --[10 s no bey]--------->  COUNTDOWN
    """

    def __init__(self, *, enabled: bool = False) -> None:
        if enabled:
            self._state = PlayState.WAITING_CONNECTION
            print("Play mode: waiting for WebSocket client...")
        else:
            self._state = PlayState.DISABLED

        self._phase: Optional[CountdownPhase] = None
        self._phase_start: float = 0.0
        self._no_track_since: Optional[float] = None
        self._no_track_timeout: float = getattr(
            config, "PLAY_NO_TRACK_TIMEOUT", 10.0
        )

    @property
    def state(self) -> PlayState:
        return self._state

    @property
    def phase(self) -> Optional[CountdownPhase]:
        return self._phase

    def is_tracking_enabled(self) -> bool:
        return self._state in (PlayState.DISABLED, PlayState.TRACKING)

    def update(self, num_tracked_beys: int, has_ws_clients: bool) -> None:
        """Advance the state machine.  Called once per frame."""
        if self._state is PlayState.DISABLED:
            return

        if self._state is PlayState.WAITING_CONNECTION:
            if has_ws_clients:
                print("Play mode: client connected -- starting countdown")
                self._start_countdown()
            return

        if self._state is PlayState.COUNTDOWN:
            self._tick_countdown()
            return

        if self._state is PlayState.TRACKING:
            now = time.time()
            if num_tracked_beys > 0:
                self._no_track_since = None
            else:
                if self._no_track_since is None:
                    self._no_track_since = now
                elif now - self._no_track_since >= self._no_track_timeout:
                    print("Play mode: no bey detected for "
                          f"{self._no_track_timeout:.0f}s -- restarting countdown")
                    self._start_countdown()

    def _start_countdown(self) -> None:
        self._state = PlayState.COUNTDOWN
        self._phase = _PHASE_ORDER[0]
        self._phase_start = time.time()
        self._no_track_since = None
        print(f"Play mode: {_PHASE_DISPLAY_TEXT[self._phase]}")

    def _tick_countdown(self) -> None:
        elapsed = time.time() - self._phase_start
        duration = _phase_duration(self._phase)
        if elapsed < duration:
            return

        idx = _PHASE_ORDER.index(self._phase)
        if idx + 1 < len(_PHASE_ORDER):
            self._phase = _PHASE_ORDER[idx + 1]
            self._phase_start = time.time()
            print(f"Play mode: {_PHASE_DISPLAY_TEXT[self._phase]}")
        else:
            self._state = PlayState.TRACKING
            self._phase = None
            self._no_track_since = None
            print("Play mode: tracking enabled")

    def get_countdown_data(self) -> dict:
        """Build the ``playMode`` sub-payload for the WebSocket message."""
        return {
            "state": self._state.value,
            "phase": self._phase.value if self._phase else None,
            "text": _PHASE_DISPLAY_TEXT.get(self._phase, ""),
            "active": self._state is PlayState.COUNTDOWN,
        }

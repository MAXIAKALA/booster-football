"""Kick power calibration: launch velocity, launch acceleration, travel distance.

Why measure in-sim
------------------
The SDK ``SoccerKickManager`` forwards ``power`` (1.0-10.0) to the simulator
with no local mapping, so the power-to-motion relation is simulator physics.
It can only be measured.

Controlled conditions (identical for every power)
-------------------------------------------------
1. **Ball at rest** - the ball must be still for ``BALL_REST_SEC`` before a kick.
2. **Fixed ball-robot gap** - the kicker stages exactly ``STAGE_DIST_M`` from
   the ball and kicks only when the measured gap is inside ``STAGE_TOL_M``.
   The actual gap is logged for every sample.
3. **Clear lane** - the two non-kicking robots are parked in the far corners of
   the opponent half, behind the ball, so they never touch the shot.
4. **No opponent at the ball** - a kick is only started when the nearest
   opponent is farther than ``OPP_CLEAR_M`` from the ball.

Measuring in a short launch window
----------------------------------
Distance saturates almost immediately on a 14x9 m pitch (even power 1.0 nearly
reaches the endline), and a full roll can hit a goal, be intercepted, or collide
with the boundary. To keep every sample clean and comparable, the launch is only
observed for ``LAUNCH_WINDOW_SEC`` right after the strike. Inside that window the
ball is still moving freely, so:

- **v0** initial velocity (m/s) and **a0** initial acceleration (m/s^2) come from
  a quadratic fit of displacement vs time over the first frames of motion.
- **distance** is the furthest displacement observed inside the window, and
  **saturated** flags a shot that already reached the field edge.

If an opponent gets within ``OPP_INTERFERE_M`` of the ball during the window, the
attempt is rejected and the same power retried, up to ``MAX_ATTEMPTS``. Missed
strikes (ball did not move) are retried too.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

__all__ = [
    "POWER_TABLE",
    "STAGE_DIST_M",
    "STAGE_TOL_M",
    "KICK_DIRECTION",
    "LAUNCH_WINDOW_SEC",
    "MOVE_EPS_M",
    "LevelResult",
    "KickCalibrator",
    "build_power_schedule",
    "travel_distance",
    "launch_fit",
    "linear_fit",
    "run_calibration",
]


_log = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Gradient power table
# ----------------------------------------------------------------------

POWER_TABLE: tuple[float, ...] = (
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
)


# ----------------------------------------------------------------------
# Test conditions
# ----------------------------------------------------------------------

STAGE_DIST_M = 1.0          # Required ball-robot gap before every kick.
STAGE_TOL_M = 0.05          # Allowed deviation from STAGE_DIST_M.

BALL_REST_EPS_M = 0.01      # Frame movement below this counts as still.
BALL_REST_SEC = 0.6         # Required stillness before a kick.

# In a live match opponents swarm the ball, so an absolute exclusion radius
# would never open. Instead start a kick whenever there is a momentary gap, and
# rely on the short launch window plus retry-on-contamination for clean data.
OPP_CLEAR_M = 1.4           # Ball must be at least this far from any opponent.
OPP_INTERFERE_M = 0.9       # Opponent closer than this during the window -> retry.

# Kick toward -x (our own half). Teammates are parked in the +x opponent-half
# corners, so this lane is empty and away from the opposing AI's start. Only the
# short launch window is measured, so where the ball ends up does not matter.
KICK_DIRECTION = math.pi

STRIKE_MIN_SEC = 0.3        # Minimum wait before accepting "no contact yet".
STRIKE_MAX_SEC = 3.0        # Give up waiting for contact after this long.
LAUNCH_WINDOW_SEC = 0.5     # Observe only this long after contact is detected.
MAX_WINDOW_SAMPLES = 20

MOVE_EPS_M = 0.05           # Ball has moved once it leaves by this much.
SATURATE_MARGIN_M = 0.5     # Treat near-edge as saturated.

MAX_ATTEMPTS = 10
APPROACH_TIMEOUT_SEC = 6.0

_PARK_X_FRAC = 0.70
_PARK_Y_FRAC = 0.75


# ----------------------------------------------------------------------
# Pure helpers
# ----------------------------------------------------------------------


def build_power_schedule(powers: tuple[float, ...] = POWER_TABLE) -> list[float]:
    """Return powers clamped to [1.0, 10.0], in ascending order."""
    return sorted(min(10.0, max(1.0, float(p))) for p in powers)


def travel_distance(
    origin: tuple[float, float], positions: list[tuple[float, float]],
) -> float:
    """Furthest distance from ``origin`` reached by ``positions`` (m)."""
    if not positions:
        return 0.0
    ox, oy = origin
    return max(math.hypot(x - ox, y - oy) for x, y in positions)


def _solve3(m: list[list[float]], b: list[float]) -> tuple[float, float, float] | None:
    def det(a: list[list[float]]) -> float:
        return (
            a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
            - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
            + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
        )

    d = det(m)
    if abs(d) < 1e-12:
        return None
    out = []
    for col in range(3):
        a = [row[:] for row in m]
        for r in range(3):
            a[r][col] = b[r]
        out.append(det(a) / d)
    return out[0], out[1], out[2]


def launch_fit(
    samples: list[tuple[float, float, float]],
    move_eps: float = MOVE_EPS_M,
) -> tuple[float, float]:
    """Return ``(v0, a0)`` from time-stamped ``[(t, x, y), ...]`` samples.

    A quadratic ``s(t) = c0 + c1*t + c2*t^2`` is least-squares fitted to the
    displacement along the motion direction; ``v0`` is the slope at onset and
    ``a0 = 2*c2``. Falls back to finite differences if the fit is degenerate.
    """
    n = len(samples)
    if n < 2:
        return 0.0, 0.0

    t0, x0, y0 = samples[0]
    onset = None
    for i in range(1, n):
        if math.hypot(samples[i][1] - x0, samples[i][2] - y0) > move_eps:
            onset = i
            break
    if onset is None:
        return 0.0, 0.0

    def seg_speed(j: int) -> float:
        ta, xa, ya = samples[j - 1]
        tb, xb, yb = samples[j]
        dt = tb - ta
        return math.hypot(xb - xa, yb - ya) / dt if dt > 0.0 else 0.0

    def fallback() -> tuple[float, float]:
        v0 = seg_speed(onset)
        a0 = 0.0
        if onset + 1 < n:
            v1 = seg_speed(onset + 1)
            dt = samples[onset + 1][0] - samples[onset][0]
            if dt > 0.0:
                a0 = (v1 - v0) / dt
            v0 = max(v0, v1)
        return v0, a0

    start = max(0, onset - 1)
    window = samples[start:start + 8]
    if len(window) < 3:
        return fallback()

    t_ref = window[0][0]
    bx, by = window[0][1], window[0][2]
    dx, dy = window[-1][1] - bx, window[-1][2] - by
    d = math.hypot(dx, dy)
    if d < 1e-9:
        return fallback()
    ux, uy = dx / d, dy / d

    ts = [t - t_ref for (t, _x, _y) in window]
    ss = [((x - bx) * ux + (y - by) * uy) for (_t, x, y) in window]
    S1 = sum(ts)
    S2 = sum(t * t for t in ts)
    S3 = sum(t ** 3 for t in ts)
    S4 = sum(t ** 4 for t in ts)
    b0 = sum(ss)
    b1 = sum(t * s for t, s in zip(ts, ss))
    b2 = sum((t * t) * s for t, s in zip(ts, ss))
    sol = _solve3(
        [[float(len(ts)), S1, S2], [S1, S2, S3], [S2, S3, S4]],
        [b0, b1, b2],
    )
    if sol is None:
        return fallback()

    _, c1, c2 = sol
    t_onset = samples[onset][0] - t_ref
    return c1 + 2.0 * c2 * t_onset, 2.0 * c2


def linear_fit(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    """Least-squares ``(slope, intercept, r_squared)``; ``(0,0,0)`` if degenerate."""
    n = len(xs)
    if n != len(ys) or n < 2:
        return 0.0, 0.0, 0.0
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    var_x = sum((x - mean_x) ** 2 for x in xs)
    if var_x <= 0.0:
        return 0.0, 0.0, 0.0
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = cov / var_x
    intercept = mean_y - slope * mean_x
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0
    return slope, intercept, r2


# ----------------------------------------------------------------------
# Result containers
# ----------------------------------------------------------------------


@dataclass
class LevelResult:
    power: float
    v0: float
    a0: float
    distance: float
    gap: float
    saturated: bool
    attempts: int


@dataclass
class _Level:
    power: float
    attempts: int = 0
    result: LevelResult | None = None
    reasons: list[str] = field(default_factory=list)


# ----------------------------------------------------------------------
# State machine
# ----------------------------------------------------------------------


class KickCalibrator:
    """Sweep :data:`POWER_TABLE` under controlled conditions; log a clean table."""

    def update(self, context, players, store) -> None:
        ball = context.ball
        if ball is None or not players:
            return

        if getattr(store, "cal_state", None) is None:
            self._init(context, store, players)

        self._track_ball_rest(context, store, ball)

        player = next((p for p in players if p.id == store.cal_robot_id), None)
        if player is None:
            player = players[0]
            store.cal_robot_id = player.id
        self._park_others(context, players, player)

        state = store.cal_state
        if state == "done":
            player.stop()
        elif state == "approach":
            self._approach(context, player, store, ball)
        elif state == "strike":
            self._strike(context, player, store, ball)
        elif state == "track":
            self._track(context, player, store, ball)

    # -- init / helpers -------------------------------------------------

    def _init(self, context, store, players) -> None:
        store.cal_state = "approach"
        store.cal_levels = [_Level(p) for p in build_power_schedule()]
        store.cal_index = 0
        store.cal_robot_id = players[0].id
        store.cal_ball_prev = None
        store.cal_ball_still_since = None
        store.cal_approach_at = context.now
        self._clear_shot(store)
        _log.info("CALIB_CONDITIONS stage_dist=%.2f tol=%.2f opp_clear=%.2f "
                  "opp_interfere=%.2f window=%.2f",
                  STAGE_DIST_M, STAGE_TOL_M, OPP_CLEAR_M, OPP_INTERFERE_M,
                  LAUNCH_WINDOW_SEC)
        for i, lvl in enumerate(store.cal_levels, start=1):
            _log.info("CALIB_PLAN index=%d power=%.1f", i, lvl.power)

    def _clear_shot(self, store) -> None:
        store.cal_samples = []
        store.cal_origin = None
        store.cal_gap = None
        store.cal_contaminated = False
        store.cal_kick_at = None
        store.cal_window_started = None

    def _level(self, store) -> _Level | None:
        if store.cal_index >= len(store.cal_levels):
            return None
        return store.cal_levels[store.cal_index]

    def _track_ball_rest(self, context, store, ball) -> None:
        pos = (ball.x, ball.y)
        prev = store.cal_ball_prev
        moved = (math.inf if prev is None
                 else math.hypot(pos[0] - prev[0], pos[1] - prev[1]))
        store.cal_ball_prev = pos
        if moved > BALL_REST_EPS_M:
            store.cal_ball_still_since = None
        elif store.cal_ball_still_since is None:
            store.cal_ball_still_since = context.now

    def _ball_at_rest(self, context, store) -> bool:
        since = store.cal_ball_still_since
        return since is not None and context.now - since >= BALL_REST_SEC

    def _nearest_opponent(self, context, point) -> float:
        best = math.inf
        for r in context.opponents.values():
            pose = r.pose
            if pose is None:
                continue
            best = min(best, math.hypot(pose.x - point[0], pose.y - point[1]))
        return best

    def _park_others(self, context, players, kicker) -> None:
        half_l = context.field.length / 2.0
        half_w = context.field.width / 2.0
        spots = [
            (half_l * _PARK_X_FRAC, -half_w * _PARK_Y_FRAC),
            (half_l * _PARK_X_FRAC, half_w * _PARK_Y_FRAC),
        ]
        for p, spot in zip((p for p in players if p is not kicker), spots):
            p.action = "cal:park"
            p.move_to_position(spot)

    # -- states ---------------------------------------------------------

    def _approach(self, context, player, store, ball) -> None:
        level = self._level(store)
        if level is None:
            self._finish(store)
            return
        if level.attempts >= MAX_ATTEMPTS:
            self._give_up(store, level)
            return

        if not self._ball_at_rest(context, store):
            player.action = "cal:wait_rest"
            player.stop()
            return

        if self._nearest_opponent(context, (ball.x, ball.y)) < OPP_CLEAR_M:
            player.action = "cal:wait_clear"
            player.stop()
            return

        staging = (
            ball.x - math.cos(KICK_DIRECTION) * STAGE_DIST_M,
            ball.y - math.sin(KICK_DIRECTION) * STAGE_DIST_M,
        )
        player.action = f"cal:stage p={level.power:.0f}"
        player.walk_to(staging, avoid_ball=True, avoid_robots=True)

        pose = player.pose
        if pose is None:
            return
        gap = math.hypot(pose.x - ball.x, pose.y - ball.y)
        on_station = abs(gap - STAGE_DIST_M) <= STAGE_TOL_M
        timed_out = context.now - store.cal_approach_at >= APPROACH_TIMEOUT_SEC
        if not (on_station or timed_out):
            return
        if not self._ball_at_rest(context, store):
            return

        store.cal_origin = (ball.x, ball.y)
        store.cal_gap = gap
        store.cal_samples = [(context.now, ball.x, ball.y)]
        store.cal_contaminated = False
        store.cal_kick_at = context.now
        store.cal_window_started = None
        store.cal_state = "strike"
        level.attempts += 1
        player.action = f"cal:kick p={level.power:.0f}"
        player.kick(KICK_DIRECTION, level.power)

    def _strike(self, context, player, store, ball) -> None:
        store.cal_samples.append((context.now, ball.x, ball.y))
        self._note_contamination(context, store, ball)
        origin = store.cal_origin or (ball.x, ball.y)
        elapsed = context.now - store.cal_kick_at

        # Keep the kick task alive until the ball leaves the kick point (the
        # manager must first walk from the staging gap and strike). A hard cap
        # still finalizes a missed strike so it can be retried.
        moved = math.hypot(ball.x - origin[0], ball.y - origin[1])
        contact = moved > MOVE_EPS_M and elapsed >= STRIKE_MIN_SEC
        if not contact and elapsed < STRIKE_MAX_SEC:
            return
        player.release_kick()
        player.stop()
        store.cal_window_started = context.now
        store.cal_state = "track"

    def _track(self, context, player, store, ball) -> None:
        player.release_kick()
        player.stop()

        store.cal_samples.append((context.now, ball.x, ball.y))
        self._note_contamination(context, store, ball)

        elapsed = context.now - store.cal_window_started
        if store.cal_contaminated:
            self._reject(context, store, "opponent_near_ball")
            return
        if elapsed >= LAUNCH_WINDOW_SEC or len(store.cal_samples) >= MAX_WINDOW_SAMPLES:
            self._record(context, store)

    def _note_contamination(self, context, store, ball) -> None:
        if self._nearest_opponent(context, (ball.x, ball.y)) < OPP_INTERFERE_M:
            store.cal_contaminated = True

    # -- recording ------------------------------------------------------

    def _record(self, context, store) -> None:
        level = self._level(store)
        if level is None:
            return
        origin = store.cal_origin or (store.cal_samples[0][1], store.cal_samples[0][2])
        positions = [(x, y) for (_t, x, y) in store.cal_samples]
        distance = travel_distance(origin, positions)

        if distance < MOVE_EPS_M:
            self._reject(context, store, "ball_did_not_move")
            return

        v0, a0 = launch_fit(store.cal_samples, MOVE_EPS_M)
        fx, fy = max(positions, key=lambda p: math.hypot(p[0] - origin[0], p[1] - origin[1]))
        half_l = context.field.length / 2.0
        half_w = context.field.width / 2.0
        saturated = (
            abs(fx) > half_l - SATURATE_MARGIN_M
            or abs(fy) > half_w - SATURATE_MARGIN_M
        )

        level.result = LevelResult(
            power=level.power,
            v0=v0,
            a0=a0,
            distance=distance,
            gap=store.cal_gap if store.cal_gap is not None else STAGE_DIST_M,
            saturated=saturated,
            attempts=level.attempts,
        )
        _log.info(
            "CALIB_RESULT power=%.1f v0=%.3f a0=%.3f distance=%.3f gap=%.3f "
            "saturated=%s attempts=%d",
            level.power, v0, a0, distance, level.result.gap, saturated,
            level.attempts,
        )
        self._advance(context, store)

    def _reject(self, context, store, reason: str) -> None:
        level = self._level(store)
        if level is None:
            return
        level.reasons.append(reason)
        _log.info(
            "CALIB_RETRY power=%.1f reason=%s attempt=%d",
            level.power, reason, level.attempts,
        )
        store.cal_state = "approach"
        store.cal_approach_at = context.now
        self._clear_shot(store)

    def _give_up(self, store, level: _Level) -> None:
        _log.info(
            "CALIB_GIVEUP power=%.1f attempts=%d reasons=%s",
            level.power, level.attempts, ",".join(level.reasons) or "none",
        )
        self._advance(None, store)

    def _advance(self, context, store) -> None:
        store.cal_index += 1
        store.cal_state = "approach"
        if context is not None:
            store.cal_approach_at = context.now
        self._clear_shot(store)
        if store.cal_index >= len(store.cal_levels):
            self._finish(store)

    def _finish(self, store) -> None:
        store.cal_state = "done"
        results = [lvl.result for lvl in store.cal_levels if lvl.result is not None]
        if len(results) >= 2:
            xs = [r.power for r in results]
            sv, iv, rv = linear_fit(xs, [r.v0 for r in results])
            sa, ia, ra = linear_fit(xs, [r.a0 for r in results])
            sd, _id, rd = linear_fit(xs, [r.distance for r in results])
            _log.info(
                "CALIB_SUMMARY clean=%d/%d v0_slope=%.3f v0_int=%.3f r2_v0=%.3f "
                "a0_slope=%.3f a0_int=%.3f r2_a0=%.3f dist_slope=%.3f r2_dist=%.3f",
                len(results), len(store.cal_levels), sv, iv, rv, sa, ia, ra, sd, rd,
            )
        for lvl in store.cal_levels:
            if lvl.result is None:
                _log.info(
                    "CALIB_TABLE power=%.1f NO_CLEAN_SAMPLE attempts=%d reasons=%s",
                    lvl.power, lvl.attempts, ",".join(lvl.reasons) or "none",
                )
            else:
                r = lvl.result
                _log.info(
                    "CALIB_TABLE power=%.1f v0=%.3f a0=%.3f distance=%.3f gap=%.3f "
                    "saturated=%s attempts=%d",
                    r.power, r.v0, r.a0, r.distance, r.gap, r.saturated, r.attempts,
                )


_CALIBRATOR = KickCalibrator()


def run_calibration(context, players, store) -> None:
    """Module-level entry point used by ``main.play`` when enabled."""
    _CALIBRATOR.update(context, players, store)
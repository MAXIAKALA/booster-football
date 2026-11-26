"""Kick power sweep at the kickoff, driven trial-by-trial.

Protocol (one trial per simulation run)
---------------------------------------
1. Stage the kicker **behind the ball on the opponent side**: the ball lies
   between the kicker and our own half, so the taker stands at the ``+x`` side
   of the ball (the side nearer the opponent half), ``STAGE_DIST_M`` away.
2. Turn to face the ball, then verify the taker is facing **our own half**. If
   it is not, the trial is aborted and the orchestrator stops the simulation
   (logged ``KEXP_FACE_ABORT``).
3. Perform a normal ``Player.kick`` with the configured power toward our half.
4. Record the ball's **launch velocity** ``v0``, **launch acceleration** ``a0``,
   **travel distance** and **travel direction**.
5. A sample is **valid** only when ``v0`` is clearly non-zero **and** the ball
   travel direction is within ``DIRECTION_TOL_DEG`` of *the direction facing our
   own half* (``OWN_HALF_DIRECTION``). Otherwise it is invalid and the same
   power is retried.
6. After every record the orchestrator stops the simulation
   (``KEXP_STOP_AFTER_RECORD``). A valid record increments the counter. When the
   counter reaches ``TARGET_VALID_SAMPLES`` valid samples, it logs
   ``KEXP_ALL_DONE`` so everything can be terminated.

Counter / power
---------------
A ``store.experiment_counter`` (initial ``1``) selects the power: ``power =
counter`` clamped to ``[KICK_POWER_MIN, KICK_POWER_MAX]`` and monotonically
climbing. Valid samples increment the counter; invalid ones do not (retry same
power). The counter lives in ``store`` so it survives match restarts inside one
running agent; re-deploying the agent resets it to ``1`` (the fresh sweep start).
"""

from __future__ import annotations

import logging
import math

from .calibration import (
    LAUNCH_WINDOW_SEC,
    MAX_WINDOW_SAMPLES,
    MOVE_EPS_M,
    launch_fit,
)
from .param import (
    ENABLE_KICKOFF_EXPERIMENT,
    KICK_POWER_MAX,
    KICK_POWER_MIN,
    KICKOFF_EXP_FIXED_POWER,
    KICKOFF_EXP_STAGE_DIST_M,
)
from .utils.geom import angle_to, normalize_angle

__all__ = ["run", "power_schedule"]

_log = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Experiment tuning
# ----------------------------------------------------------------------

KICK_DIRECTION = math.pi        # Toward our own half (-x).
PREP_OFFSET_X = -0.6            # READY prep point on our own side.
GO_AROUND_LATERAL_M = 0.9       # Waypoint used to circle the ball.

STAGE_TOL_M = 0.10              # Allowed |dx - STAGE_DIST_M| when on station.
LATERAL_TOL_M = 0.35            # Allowed |dy| when on station.
HEADING_TOL_RAD = 0.15          # Facing-the-ball tolerance before kicking.
FACE_HALF_TOL_RAD = math.radians(75.0)  # Facing-own-half tolerance (abort if worse)
DIRECTION_TOL_DEG = 60.0        # Ball direction vs facing tolerance for validity.
MIN_LAUNCH_SPEED_MPS = 0.10     # Below this launch speed a sample is invalid.

BALL_REST_EPS_M = 0.01
BALL_REST_SEC = 0.6
STRIKE_MIN_SEC = 0.3
STRIKE_MAX_SEC = 5.0
APPROACH_TIMEOUT_SEC = 6.0
MAX_TRACK_SEC = 6.0
BALL_STOP_SPEED = 0.10
BALL_STOP_SEC = 0.4
MAX_SAMPLES = MAX_WINDOW_SAMPLES * 4

TARGET_VALID_SAMPLES = 10
MAX_ATTEMPTS_PER_POWER = 6      # Safety cap so a stuck power cannot loop forever.

# Direction our own half is declared on (world/field frame, our-half = -x).
OWN_HALF_DIRECTION = math.pi


# ----------------------------------------------------------------------
# Power schedule derived from the counter
# ----------------------------------------------------------------------

def power_schedule() -> list[float]:
    """Ascending power levels between KICK_POWER_MIN and KICK_POWER_MAX (step 1)."""
    levels: list[float] = []
    power = float(KICK_POWER_MIN)
    while power <= KICK_POWER_MAX + 1e-9:
        levels.append(round(power, 3))
        power += 1.0
    return levels or [float(KICK_POWER_MIN)]


def _power_for_counter(counter: int) -> float:
    # A fixed power (if configured) overrides the counter-based schedule, so the
    # same power is used for every trial (handy for a single watchable run).
    if KICKOFF_EXP_FIXED_POWER is not None:
        return max(KICK_POWER_MIN, min(KICK_POWER_MAX, float(KICKOFF_EXP_FIXED_POWER)))
    levels = power_schedule()
    index = max(0, min(counter - 1, len(levels) - 1))
    return levels[index]


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------

def run(context, players, store, phase) -> None:
    """Drive one kickoff trial for this frame."""
    if not ENABLE_KICKOFF_EXPERIMENT or not players:
        return

    ball = context.ball
    phase_name = getattr(phase, "name", "")

    # Initialise persistent counter and per-trial state on first use.
    if getattr(store, "experiment_counter", None) is None:
        store.experiment_counter = 1
    if getattr(store, "experiment_state", None) is None:
        _init_trial(store, players)

    # Once a trial is finished, a fresh READY (new match) re-arms the next trial.
    if getattr(store, "experiment_state", None) == "finished" and phase_name == "READY":
        _init_trial(store, players)

    kicker = next(
        (p for p in players if p.id == store.experiment_robot_id), players[0]
    )
    store.experiment_robot_id = kicker.id

    # All non-kicker robots stay idle.
    for p in players:
        if p is not kicker:
            p.action = "kexp:idle"
            p.stop()

    if ball is None:
        kicker.action = "kexp:no_ball"
        kicker.stop()
        return

    _track_ball_rest(context, store, ball)

    state = store.experiment_state
    if state == "waiting":
        _waiting(context, store, kicker, ball, phase_name)
    elif state == "strike":
        _strike(context, store, kicker, ball)
    elif state == "track":
        _track(context, store, kicker, ball)
    else:  # "finished"
        kicker.action = "kexp:finished"
        kicker.stop()


# ----------------------------------------------------------------------
# Per-trial state
# ----------------------------------------------------------------------

def _init_trial(store, players) -> None:
    # Pick the teammate closest to the centre spot as the taker.
    def key(p):
        pose = p.pose
        return math.inf if pose is None else math.hypot(pose.x, pose.y)

    kicker = min(players, key=key)

    if getattr(store, "experiment_counter", None) is None:
        store.experiment_counter = 1

    store.experiment_state = "waiting"
    store.experiment_robot_id = kicker.id
    store.experiment_power = _power_for_counter(store.experiment_counter)
    store.experiment_attempts = 0
    store.experiment_rounded = False
    store.experiment_committed = False
    store.experiment_stage_at = None
    store.experiment_ball_prev = None
    store.experiment_rest_since = None
    store.experiment_origin = None
    store.experiment_samples = []
    store.experiment_gap = None
    store.experiment_kick_at = None
    store.experiment_window_started = None
    store.experiment_still_since = None
    store.experiment_face_heading = None
    _log.info(
        "KEXP_TRIAL counter=%d power=%.1f kicker=%s",
        store.experiment_counter, store.experiment_power, kicker.id,
    )


def _track_ball_rest(context, store, ball) -> None:
    pos = (ball.x, ball.y)
    prev = store.experiment_ball_prev
    moved = (
        math.inf if prev is None
        else math.hypot(pos[0] - prev[0], pos[1] - prev[1])
    )
    store.experiment_ball_prev = pos
    if moved > BALL_REST_EPS_M:
        store.experiment_rest_since = None
    elif store.experiment_rest_since is None:
        store.experiment_rest_since = context.now


def _ball_resting(context, store) -> bool:
    since = store.experiment_rest_since
    return since is not None and context.now - since >= BALL_REST_SEC


# ----------------------------------------------------------------------
# State machine
# ----------------------------------------------------------------------

def _waiting(context, store, kicker, ball, phase_name: str) -> None:
    if phase_name == "READY":
        # Hold on our own side until the kickoff is live.
        kicker.action = "kexp:prep"
        kicker.walk_to(
            (ball.x + PREP_OFFSET_X, ball.y),
            face=KICK_DIRECTION,
            avoid_ball=True,
        )
        return

    committed = getattr(store, "experiment_committed", False)
    can_start = phase_name in ("OUR_KICKOFF", "NORMAL")
    if not committed and not can_start:
        kicker.action = "kexp:wait"
        kicker.stop()
        return

    store.experiment_committed = True
    _stage(context, store, kicker, ball)


def _stage(context, store, kicker, ball) -> None:
    """Circle to the +x side of the ball, stand behind it, face the ball."""
    pose = kicker.pose
    if pose is None:
        kicker.stop()
        return
    bx, by = ball.x, ball.y

    # 1. Circle around the ball via a lateral waypoint (avoid pushing it).
    if not store.experiment_rounded:
        side = (bx, by + GO_AROUND_LATERAL_M)
        kicker.action = "kexp:round"
        reached = kicker.walk_to(side, face=KICK_DIRECTION, avoid_ball=True)
        if reached or (pose.y - by) >= GO_AROUND_LATERAL_M - 0.25:
            store.experiment_rounded = True
            store.experiment_stage_at = context.now
        return

    # 2. Approach the point directly behind the ball on +x, facing the ball.
    target = (bx + KICKOFF_EXP_STAGE_DIST_M, by)
    face_ball = angle_to(pose.x, pose.y, bx, by)
    kicker.action = f"kexp:stage p={store.experiment_power:.1f}"
    kicker.walk_to(target, face=face_ball)

    pose = kicker.pose
    if pose is None:
        return
    dx = pose.x - bx
    dy = pose.y - by
    on_station = (
        abs(dx - KICKOFF_EXP_STAGE_DIST_M) <= STAGE_TOL_M
        and abs(dy) <= LATERAL_TOL_M
    )
    timed_out = (
        store.experiment_stage_at is not None
        and context.now - store.experiment_stage_at >= APPROACH_TIMEOUT_SEC
    )
    behind = dx >= KICKOFF_EXP_STAGE_DIST_M * 0.5
    if not (behind and (on_station or timed_out)):
        return
    if not _ball_resting(context, store):
        return

    # 3. Face the ball; wait until aligned.
    face_err = abs(normalize_angle(angle_to(pose.x, pose.y, bx, by) - pose.theta))
    if face_err > HEADING_TOL_RAD:
        kicker.action = "kexp:align"
        kicker.walk_to((pose.x, pose.y), face=angle_to(pose.x, pose.y, bx, by))
        return

    # 4. Verify the taker faces our own half. If not, abort the trial so the
    #    orchestrator can stop the simulation.
    half_err = abs(normalize_angle(pose.theta - OWN_HALF_DIRECTION))
    if half_err > FACE_HALF_TOL_RAD:
        _abort_face(store, pose.theta, half_err)
        kicker.action = "kexp:face_abort"
        kicker.stop()
        return

    store.experiment_face_heading = pose.theta
    _begin_strike(context, store, kicker, ball, math.hypot(dx, dy))


def _abort_face(store, heading: float, half_err: float) -> None:
    store.experiment_state = "finished"
    _log.info(
        "KEXP_FACE_ABORT counter=%d power=%.1f heading=%.3f half_err_deg=%.1f",
        store.experiment_counter, store.experiment_power, heading,
        math.degrees(half_err),
    )
    _log.info("KEXP_STOP_AFTER_RECORD counter=%d power=%.1f",
              store.experiment_counter, store.experiment_power)


def _begin_strike(context, store, kicker, ball, gap: float) -> None:
    store.experiment_origin = (ball.x, ball.y)
    store.experiment_gap = gap
    store.experiment_samples = [(context.now, ball.x, ball.y)]
    store.experiment_kick_at = context.now
    store.experiment_window_started = None
    store.experiment_still_since = None
    store.experiment_state = "strike"
    kicker.action = f"kexp:kick p={store.experiment_power:.1f}"
    kicker.kick(KICK_DIRECTION, store.experiment_power)


def _strike(context, store, kicker, ball) -> None:
    store.experiment_samples.append((context.now, ball.x, ball.y))
    origin = store.experiment_origin or (ball.x, ball.y)
    elapsed = context.now - store.experiment_kick_at
    moved = math.hypot(ball.x - origin[0], ball.y - origin[1])
    contact = moved > MOVE_EPS_M and elapsed >= STRIKE_MIN_SEC
    if not contact and elapsed < STRIKE_MAX_SEC:
        # Keep the kick task alive and refreshed with the current ball position
        # so the manager keeps walking in to strike until contact.
        kicker.action = f"kexp:strike p={store.experiment_power:.1f}"
        kicker.kick(KICK_DIRECTION, store.experiment_power)
        return
    kicker.release_kick()
    kicker.stop()
    store.experiment_window_started = context.now
    store.experiment_state = "track"


def _track(context, store, kicker, ball) -> None:
    kicker.release_kick()
    kicker.stop()
    store.experiment_samples.append((context.now, ball.x, ball.y))

    speed = _ball_speed(store.experiment_samples)
    if speed < BALL_STOP_SPEED:
        if store.experiment_still_since is None:
            store.experiment_still_since = context.now
    else:
        store.experiment_still_since = None

    since_window = context.now - (store.experiment_window_started or context.now)
    since_kick = context.now - (store.experiment_kick_at or context.now)
    rested = (
        store.experiment_still_since is not None
        and context.now - store.experiment_still_since >= BALL_STOP_SEC
    )
    launch_measured = since_window >= LAUNCH_WINDOW_SEC
    hit_cap = (
        since_kick >= MAX_TRACK_SEC
        or len(store.experiment_samples) >= MAX_SAMPLES
    )
    if hit_cap or (launch_measured and rested):
        _record(context, store)


def _ball_speed(samples: list[tuple[float, float, float]]) -> float:
    if len(samples) < 2:
        return math.inf
    t0, x0, y0 = samples[-2]
    t1, x1, y1 = samples[-1]
    dt = t1 - t0
    if dt <= 0.0:
        return math.inf
    return math.hypot(x1 - x0, y1 - y0) / dt


# ----------------------------------------------------------------------
# Recording + validity + counter
# ----------------------------------------------------------------------

def _path_and_direction(
    origin: tuple[float, float], samples: list[tuple[float, float, float]],
) -> tuple[float, float | None]:
    """Return (total path length, net displacement direction in radians)."""
    total = 0.0
    px, py = origin
    far_x, far_y = origin
    far_d = 0.0
    for _t, x, y in samples:
        total += math.hypot(x - px, y - py)
        px, py = x, y
        d = math.hypot(x - origin[0], y - origin[1])
        if d > far_d:
            far_d = d
            far_x, far_y = x, y
    if far_d < 1e-6:
        return total, None
    return total, angle_to(origin[0], origin[1], far_x, far_y)


def _record(context, store) -> None:
    samples = store.experiment_samples
    origin = store.experiment_origin or (
        (samples[0][1], samples[0][2]) if samples else (0.0, 0.0)
    )
    distance, direction = _path_and_direction(origin, samples)
    v0, a0 = launch_fit(samples, MOVE_EPS_M)

    # Validity is measured against "the direction facing our own half", not the
    # exact chassis heading: the taker already passed the face-own-half check.
    if direction is None:
        dir_err_deg = 180.0
    else:
        dir_err_deg = abs(
            math.degrees(normalize_angle(direction - OWN_HALF_DIRECTION))
        )

    speed_ok = v0 > MIN_LAUNCH_SPEED_MPS
    dir_ok = dir_err_deg <= DIRECTION_TOL_DEG
    valid = speed_ok and dir_ok

    direction_deg = None if direction is None else math.degrees(direction)

    _log.info(
        "KEXP_RESULT counter=%d power=%.1f v0=%.3f a0=%.3f distance=%.3f "
        "dir_deg=%s dir_err_deg=%.1f speed_ok=%s dir_ok=%s valid=%s "
        "gap=%.3f samples=%d",
        store.experiment_counter, store.experiment_power, v0, a0, distance,
        "nan" if direction_deg is None else f"{direction_deg:.1f}",
        dir_err_deg, speed_ok, dir_ok, valid,
        store.experiment_gap if store.experiment_gap is not None else float("nan"),
        len(samples),
    )

    # Counter logic.
    if valid:
        store.experiment_counter += 1
        if store.experiment_counter > TARGET_VALID_SAMPLES:
            _log.info("KEXP_ALL_DONE valid_samples=%d", TARGET_VALID_SAMPLES)
        else:
            _log.info(
                "KEXP_VALID counter_next=%d next_power=%.1f",
                store.experiment_counter,
                _power_for_counter(store.experiment_counter),
            )
    else:
        store.experiment_attempts += 1
        if store.experiment_attempts >= MAX_ATTEMPTS_PER_POWER:
            # Avoid a permanently stuck power: move on without counting it.
            store.experiment_counter += 1
            _log.info(
                "KEXP_SKIP power=%.1f attempts=%d -> next counter=%d",
                store.experiment_power, store.experiment_attempts,
                store.experiment_counter,
            )
        else:
            _log.info(
                "KEXP_INVALID power=%.1f reason=%s",
                store.experiment_power,
                "speed" if not speed_ok else "direction",
            )

    store.experiment_state = "finished"
    _log.info("KEXP_STOP_AFTER_RECORD counter=%d power=%.1f",
              store.experiment_counter, store.experiment_power)
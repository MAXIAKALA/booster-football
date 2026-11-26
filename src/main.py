"""SoccerSim strategy entry point and primary match logic.

Structure, from high level to low level:
- main.py (this file): match strategy. ``play()`` dispatches through the Phase
  state machine to ``_act_*`` functions. Each function selects an attacker,
  usually the player nearest the ball, and invokes Player actions directly.
- player.py: Player control handle and high-level actions such as ``attack``,
  ``take_kickoff``, ``move_to_position``, and ``walk_to``. Add new skills there.
- utils/: movement, geometry, and obstacle-avoidance helpers such as
  ``opponent_goal``, ``dist``, and ``angle_to``.
- framework/: platform plumbing that users normally do not modify.

To change the playing style, primarily edit the Phase state machine, ``_act_*``
behaviors, and positioning formulas in this file.
"""

from __future__ import annotations

import logging
import math
from enum import Enum

from booster_agent_framework import AgentBase

from .calibration import run_calibration
from . import kickoff_experiment
from .framework.agent import SoccerAgentMixin
from .framework.types import KICKING_TEAM_NONE, Context, GameState, SetPlay
from .param import *
from .player import Player
from .utils import (
    angle_to,
    clamp,
    clamp_inside_field,
    dist,
    normalize_angle,
    opponent_goal,
    own_goal,
)


_log = logging.getLogger(__name__)


# ======================================================================
# Phase state machine: match phase classification
# ======================================================================


class Phase(Enum):
    """Top-level match phase controlling normal play, restarts, and stops."""
    NORMAL = "normal"              # Normal contest while PLAYING
    OUR_KICKOFF = "our_kickoff"    # Our kickoff during early SET/PLAYING
    OPP_KICKOFF = "opp_kickoff"    # Opponent kickoff; keep clear
    OUR_SET_PLAY = "our_set_play"  # Our free kick, corner, or goal kick
    OPP_SET_PLAY = "opp_set_play"  # Opponent set play; keep clear
    READY = "ready"                # Move into READY positions
    STOPPED = "stopped"            # Non-kickoff SET, INITIAL, FINISHED, or stopped


def get_phase(context: Context) -> Phase:
    """Determine the current phase from the GameController state."""
    g = context.game
    if g is None:
        return Phase.STOPPED

    state = g.state

    # READY: move to ready positions.
    if state == GameState.READY:
        return Phase.READY

    # PLAYING: normal play or an active kickoff/set play.
    if state == GameState.PLAYING and not g.stopped:
        # For a set play, kicking_team identifies the team taking it.
        if g.set_play != SetPlay.NONE and g.kicking_team != KICKING_TEAM_NONE:
            our_team = context.team_id
            if g.kicking_team == our_team:
                return Phase.OUR_SET_PLAY
            else:
                return Phase.OPP_SET_PLAY

        # During the kickoff countdown, kicking_team identifies the taker.
        if g.secondary_time > 0 and g.kicking_team != KICKING_TEAM_NONE:
            our_team = context.team_id
            if g.kicking_team == our_team:
                return Phase.OUR_KICKOFF
            else:
                return Phase.OPP_KICKOFF

        # Normal play.
        return Phase.NORMAL

    # Remain stationary during SET, INITIAL, FINISHED, or a stoppage.
    return Phase.STOPPED

def get_set_play_type(context: Context) -> SetPlay:
    """Return the active set-play type, or ``SetPlay.NONE`` when unavailable.

    This reads the GameController's ``set_play`` field without identifying the
    taking team. :func:`get_phase` distinguishes OUR_SET_PLAY from OPP_SET_PLAY;
    this function only identifies the type.

    The seven possible values from framework.types.SetPlay are:
    - ``NONE``: no set play, such as normal play or a kickoff
    - ``DIRECT_FREE_KICK``: a direct free kick that may score immediately
    - ``INDIRECT_FREE_KICK``: another player must touch the ball before a goal
    - ``PENALTY_KICK``: a penalty kick
    - ``THROW_IN``: a kick-in from touch
    - ``GOAL_KICK``: a goal kick
    - ``CORNER_KICK``: a corner kick
    """
    g = context.game
    if g is None:
        return SetPlay.NONE
    return g.set_play


# ======================================================================
# Agent entry point
# ======================================================================


class SoccerSimAgent(SoccerAgentMixin, AgentBase):
    """3v3 SoccerSim agent."""

    player_class = Player

    def init_store(self, store) -> None:
        _log.info("init_store called")
        store.prev_phase = None       # Previous phase, used to detect transitions
        store.cur_phase = None
        store.kickoff_taker = None    # Locked taker ID, reselected for each kickoff
        store.normal_attacker = None

        # Kickoff-route memory: a scripted backward kickoff that persists into
        # open play until the ball is advanced or the route expires.
        store.kickoff_route_active = False
        store.kickoff_started_at = None
        store.kickoff_ball_played = False   # Backward kick has sent the ball away
        store.kickoff_advanced = False      # Kicker has broken forward
        store.kickoff_aux_at_slot = False   # Auxiliary has reached slot 3
        store.kickoff_receiver = None
        store.kickoff_aux = None
        store.kickoff_holder = None
        store.kickoff_lineup_ready = False   # All players on their kickoff spots
        store.shot_tick = 0                  # Frame counter for the global shot rule
        store.shot_last_prob = 0.0
        store.shot_taken_at = None

        # Attack: holder-pass evaluation counter and the last pass time.
        store.attack_tick = 0
        store.attack_passed_at = None

        # Defense: ID of the player currently assigned as goalkeeper (or None).
        store.defense_keeper = None

        # Set play: per-receiver consecutive feasible-pass counters and the
        # start time used by the wait timeout.
        store.setplay_started_at = None
        store.setplay_ticks = {}

    @staticmethod
    def play(context: Context, players: list[Player], store) -> None:
        phase = get_phase(context)
        store.prev_phase = store.cur_phase
        store.cur_phase = phase

        # The scripted kickoff route only survives into open play; any other
        # phase (set play, opponent restart, READY, STOPPED) ends it cleanly.
        if phase not in (Phase.NORMAL, Phase.OUR_KICKOFF):
            store.kickoff_route_active = False

        # Set-play counters and the wait timer belong to one set play, so clear
        # them whenever the phase changes.
        if store.prev_phase != phase:
            store.setplay_started_at = None
            store.setplay_ticks = {}

        # Draw per-frame visualizations.
        _analyze_and_draw(context, players, store)

        # Draw the current phase as a label just outside the field.
        from .framework import debugdraw
        g = context.game
        game_state = g.state.value if g is not None else "none"
        set_play = g.set_play.value if g is not None else "none"
        secondary_time = g.secondary_time if g is not None else 0.0
        debugdraw.text(
            0.0, context.field.width / 2.0 + 0.2,
            f"phase={phase.value} state={game_state} set={set_play} secondary={secondary_time:.1f}",
            rgb=(1.0, 1.0, 0.0), ns="phase",
        )

        # Handle readiness and retain only players that can act this frame.
        # ensure_ready asynchronously gets up or switches to walk mode without
        # moving. Penalized players also prepare so they can rejoin immediately.
        # Penalized or unready players are excluded from role assignment to avoid
        # selecting an attacker who cannot move.
        active: list[Player] = []
        for p in players:
            ready = p.ensure_ready()
            if p.is_penalized:
                p.action = "penalized"     # May prepare, but cannot move
                p.stop()
            elif not ready:
                p.action = "fallen" if p.is_fallen else "switching_mode"
            elif p.pose is None:
                p.action = "no_pose"       # Exclude players with unknown positions
                p.stop()
            else:
                active.append(p)

        # Temporary kickoff power-sweep experiment. The kicker stages behind the
        # ball on the opponent side, faces our own half, kicks toward our half
        # with the counter-selected power, records v0/a0/distance/direction, then
        # signals a stop. Off unless ENABLE_KICKOFF_EXPERIMENT.
        if ENABLE_KICKOFF_EXPERIMENT:
            kickoff_experiment.run(context, active, store, phase)
            for p in players:
                _draw_teammate_marker(p)
            return

        # Temporary kick power→distance sweep. Runs only during free play so it
        # never violates stationary/ready rules; logs CALIB_* lines and bypasses
        # the match strategy while enabled. Disable via ENABLE_KICK_CALIBRATION.
        if ENABLE_KICK_CALIBRATION and phase == Phase.NORMAL:
            run_calibration(context, active, store)
            for p in players:
                _draw_teammate_marker(p)
            return

        # Global shot rule: every SHOT_CHECK_INTERVAL_TICKS frames, evaluate the
        # shot and fire directly when the goal probability is high enough.
        if phase in (Phase.NORMAL, Phase.OUR_KICKOFF, Phase.OUR_SET_PLAY):
            if _global_shot_rule(context, active, store) is not None:
                for p in players:
                    _draw_teammate_marker(p)
                return

        # Dispatch the team once per phase; _act_* owns team-wide role assignment.
        if phase == Phase.NORMAL:
            # Keep running the scripted kickoff route into open play until it
            # finishes or possession is lost; otherwise fall back to normal.
            if not _continue_kickoff_route(context, active, store):
                store.kickoff_route_active = False
                _act_normal(context, active, store)
        elif phase == Phase.OUR_KICKOFF:
            _clear_normal_sticky(store)
            _act_our_kickoff(context, active, store)
        elif phase == Phase.OPP_KICKOFF:
            _clear_normal_sticky(store)
            _act_opp_kickoff(context, active)
        elif phase == Phase.OUR_SET_PLAY:
            _clear_normal_sticky(store)
            _act_our_set_play(context, active, store)
        elif phase == Phase.OPP_SET_PLAY:
            _clear_normal_sticky(store)
            _act_opp_set_play(context, active, store)
        elif phase == Phase.READY:
            _clear_normal_sticky(store)
            _act_ready(context, active)
        elif phase == Phase.STOPPED:
            _clear_normal_sticky(store)
            for p in active:
                p.action = "stopped"
                p.stop()

        # Draw every teammate last, including penalized, unready, and stopped
        # players, so markers remain visible during states such as SET.
        for p in players:
            _draw_teammate_marker(p)


def _clear_normal_sticky(store) -> None:
    store.normal_attacker = None


def _player_dist_to_position(p: Player, target: tuple[float, float]) -> float:
    """Return infinity when the player's position is unknown."""
    pose = p.pose
    if pose is None:
        return math.inf
    return dist(pose.x, pose.y, *target)


def _player_dist_to_ball(context: Context, p: Player) -> float:
    """Return a player's distance to the ball's current position."""
    ball = context.ball
    return (
        _player_dist_to_position(p, (ball.x, ball.y)) + _fallen_time_cost(p)
        if ball is not None else math.inf
    )


def _fallen_time_cost(p: Player) -> float:
    return FALLEN_COST if p.is_fallen else 0.0


def _select_closest_attacker(
    context: Context,
    players: list[Player],
    preferred_id: int | None = None,
) -> Player:
    """Select the player with the lowest effective distance to the ball.

    ``players`` must be nonempty, ready, and have known poses. Normal play and
    kickoffs share this selection logic.
    """
    ranked = [(p, _player_dist_to_ball(context, p)) for p in players]
    best, best_dist = min(ranked, key=lambda item: item[1])
    preferred = next((item for item in ranked if item[0].id == preferred_id), None)
    if (
        preferred is not None
        and preferred[1] <= best_dist + ATTACKER_KEEP_DIST_MARGIN_M
    ):
        return preferred[0]
    return best


def _act_normal(context: Context, players: list[Player], store) -> None:
    """Attack with the nearest player and defend with everyone else.

    Attacking never assigns a goalkeeper: the nearest player holds the ball
    (or passes it when shooting is poor and a teammate is better placed) while
    the rest press toward the ball. A goalkeeper only appears through
    ``_act_defend`` when our penalty area is threatened.

    ``players`` contains ready players with known poses.
    """
    if not players:
        return

    attacker = _select_closest_attacker(
        context, players, getattr(store, "normal_attacker", None),
    )
    store.normal_attacker = attacker.id

    # Attack: no goalkeeper. The holder may pass when its shot is poor and a
    # teammate is in a much better receiving position.
    if _attack_pass_rule(context, attacker, players, store):
        _act_defend(context, players, store, attacker)
        return

    attacker.action = "attack"
    attacker.attack()
    _act_defend(context, players, store, attacker)


def _in_our_penalty_area(context: Context, x: float, y: float) -> bool:
    """Return whether ``(x, y)`` lies inside our own penalty area."""
    f = context.field
    return (
        x <= -f.length / 2.0 + f.penalty_area_length
        and abs(y) <= f.penalty_area_width / 2.0
    )


def _box_threatened(context: Context) -> bool:
    """Return whether the ball or the opponent on the ball is in our box."""
    ball = context.ball
    if ball is None:
        return False
    if _in_our_penalty_area(context, ball.x, ball.y):
        return True
    opp = _nearest_opponent_pose(context, (ball.x, ball.y))
    if opp is not None and _in_our_penalty_area(context, opp.x, opp.y):
        return True
    return False


def _act_defend(
    context: Context, players: list[Player], store, attacker: Player | None,
) -> None:
    """Defend with high pressing; keep a keeper only when the box is at risk.

    Outside our penalty area no keeper is assigned and every other player runs
    straight at the ball. Once the ball or the opponent on the ball enters the
    box, the player nearest our goal becomes the goalkeeper and covers the
    threat-to-goal line, while the rest keep chasing the ball.
    """
    rest = [p for p in players if p is not attacker]
    if not rest:
        return

    if _box_threatened(context):
        gx, gy = own_goal(context)
        keeper = min(rest, key=lambda p: _player_dist_to_position(p, (gx, gy)))
        store.defense_keeper = keeper.id
        keeper.action = "keeper"
        keeper.guard_goal_line()
        # Non-keepers always move straight toward the ball.
        for p in rest:
            if p is keeper:
                continue
            p.action = "press"
            p.chase_ball()
        return

    # No keeper: every non-holder defender moves straight toward the ball.
    store.defense_keeper = None
    for p in rest:
        p.action = "press"
        p.chase_ball()


def _receiving_favour(
    context: Context, receiver: Player, holder: Player | None = None,
) -> float:
    """Score how favourable it is to pass to ``receiver`` in [0, 1].

    Three dimensions, matching the requested criteria:
    - closeness to the opponent goal (``RECEIVE_W_GOAL``),
    - open space around the receiver, i.e. distance to the nearest defender
      (``RECEIVE_W_SPACE``),
    - how close the ball is to the receiver-to-goal line (``RECEIVE_W_LINE``).
    """
    ctx = context
    pose = receiver.pose
    ball = ctx.ball
    if pose is None or ball is None:
        return 0.0

    f = ctx.field
    goal_x = f.length / 2.0

    # 1) Goal proximity: 1.0 at the goal line, 0.0 at our own goal line.
    goal_term = clamp((pose.x + goal_x) / max(f.length, 1e-6), 0.0, 1.0)

    # 2) Space: nearest defender distance saturating at 2.5 m.
    d_def = _nearest_opponent_dist_to_point(ctx, (pose.x, pose.y))
    space_term = 1.0 if d_def is None else clamp(d_def / 2.5, 0.0, 1.0)

    # 3) Ball near the receiver-to-goal line (perpendicular offset).
    vx, vy = goal_x - pose.x, 0.0 - pose.y
    length2 = vx * vx + vy * vy
    if length2 < 1e-6:
        line_term = 0.0
    else:
        t = clamp(((ball.x - pose.x) * vx + (ball.y - pose.y) * vy) / length2, 0.0, 1.0)
        px, py = pose.x + vx * t, pose.y + vy * t
        offset = dist(ball.x, ball.y, px, py)
        line_term = clamp(1.0 - offset / 3.0, 0.0, 1.0)

    score = (
        RECEIVE_W_GOAL * goal_term
        + RECEIVE_W_SPACE * space_term
        + RECEIVE_W_LINE * line_term
    )
    # A pass that cannot be completed cleanly is not favourable.
    pass_prob = _pass_feasibility(ctx, ball, (pose.x, pose.y))
    return clamp(score * clamp(pass_prob / max(PASS_CORRIDOR_M, 1e-6), 0.0, 1.0), 0.0, 1.0)


def _best_receiving_teammate(
    context: Context, holder: Player, players: list[Player],
) -> tuple[Player | None, float]:
    """Return the teammate with the highest receiving favour and its score."""
    best: Player | None = None
    best_score = 0.0
    for t in players:
        if t is holder or t.pose is None:
            continue
        score = _receiving_favour(context, t, holder)
        if score > best_score:
            best_score, best = score, t
    return best, best_score


def _attack_pass_rule(
    context: Context, holder: Player, players: list[Player], store,
) -> bool:
    """Every ATTACK_EVAL_TICKS cycles, pass when shooting is poor.

    Returns True when this frame ordered a pass, so the caller skips attacking.
    The shot must be below ``ATTACK_SHOT_PROB_THRESHOLD`` and some teammate's
    receiving favour must exceed ``ATTACK_RECEIVE_PROB_THRESHOLD``.
    """
    store.attack_tick = (getattr(store, "attack_tick", 0) + 1) % max(
        1, ATTACK_EVAL_TICKS
    )
    if store.attack_tick != 0:
        return False
    ball = context.ball
    if ball is None or holder.pose is None:
        return False

    _aim, shot_prob = _shot_goal_probability(context, ball)
    if shot_prob >= ATTACK_SHOT_PROB_THRESHOLD:
        return False

    mate, favour = _best_receiving_teammate(context, holder, players)
    if mate is None or mate.pose is None or favour < ATTACK_RECEIVE_PROB_THRESHOLD:
        return False

    pose = mate.pose
    d = dist(ball.x, ball.y, pose.x, pose.y)
    power = min(KICK_POWER_PASS_MAX, KICK_POWER_PASS_BASE + KICK_POWER_PASS_PER_M * d)
    # The receiver is notified so it turns toward the ball to trap the pass.
    _notify_pass_receiver(context, players, mate.id)
    mate.turn_to_receive()
    holder.action = f"pass{mate.id}"
    holder.strike_toward((pose.x, pose.y), power)
    store.attack_passed_at = context.now
    return True


def _clamp_slot(context: Context, point: tuple[float, float]) -> tuple[float, float]:
    """Clamp a tactical slot inside the pitch with the kickoff margin."""
    f = context.field
    m = KICKOFF_FIELD_MARGIN_M
    hl = f.length / 2.0 - m
    hw = f.width / 2.0 - m
    return (clamp(point[0], -hl, hl), clamp(point[1], -hw, hw))


def _kickoff_slots(context: Context) -> tuple[
    tuple[float, float], tuple[float, float], tuple[float, float]
]:
    """Return ``(slot1, slot2, slot3)`` = (kicker, receiver, auxiliary).

    Values come from the reference snapshots of the first tactical point after
    the kick: slot1 ``(0.93, 1.81)``, slot2 ``(-1.40, 0.26)`` and slot3
    ``(1.24, -1.96)``, all team-relative with +x toward the opponent goal.
    """
    return (
        _clamp_slot(context, KICKOFF_SLOT_KICKER),
        _clamp_slot(context, KICKOFF_SLOT_RECEIVER),
        _clamp_slot(context, KICKOFF_SLOT_AUX),
    )


def _nearest_opponent_dist_to_point(context: Context, point) -> float | None:
    """Return the nearest visible opponent's distance to ``point``, or None."""
    best: float | None = None
    for r in context.opponents.values():
        pose = r.pose
        if pose is None:
            continue
        d = dist(pose.x, pose.y, point[0], point[1])
        if best is None or d < best:
            best = d
    return best


def _nearest_opponent_pose(context: Context, point):
    """Return the nearest visible opponent pose to ``point``, or None."""
    best = None
    best_d = math.inf
    for r in context.opponents.values():
        op = r.pose
        if op is None:
            continue
        d = dist(op.x, op.y, point[0], point[1])
        if d < best_d:
            best_d, best = d, op
    return best


def _screen_target(context: Context, pose, slot: tuple[float, float]):
    """Bend a slot run slightly toward the nearest defender (挡拆 effect)."""
    if pose is None:
        return slot
    op = _nearest_opponent_pose(context, (pose.x, pose.y))
    if op is None:
        return slot
    ux, uy = op.x - pose.x, op.y - pose.y
    n = math.hypot(ux, uy)
    if n < 1e-6:
        return slot
    return (
        slot[0] + ux / n * KICKOFF_KICKER_SCREEN_M,
        slot[1] + uy / n * KICKOFF_KICKER_SCREEN_M,
    )


def _aux_avoid_target(context: Context, pose, slot: tuple[float, float]):
    """Nudge the auxiliary slot away from a defender standing on its path."""
    if pose is None:
        return slot
    op = _nearest_opponent_pose(context, (pose.x, pose.y))
    if op is None:
        return slot
    if dist(pose.x, pose.y, op.x, op.y) >= KICKOFF_AUX_AVOID_M:
        return slot
    ux, uy = pose.x - op.x, pose.y - op.y
    n = math.hypot(ux, uy)
    if n < 1e-6:
        return slot
    return (
        slot[0] + ux / n * KICKOFF_AUX_SIDESTEP_M,
        slot[1] + uy / n * KICKOFF_AUX_SIDESTEP_M,
    )


def _segment_clearance(
    context: Context, ax: float, ay: float, bx: float, by: float,
) -> float:
    """Minimum lateral distance from any opponent to segment a→b, or inf."""
    vx, vy = bx - ax, by - ay
    length2 = vx * vx + vy * vy
    best = math.inf
    for r in context.opponents.values():
        op = r.pose
        if op is None:
            continue
        if length2 < 1e-6:
            d = dist(op.x, op.y, ax, ay)
        else:
            t = clamp(((op.x - ax) * vx + (op.y - ay) * vy) / length2, 0.0, 1.0)
            px, py = ax + vx * t, ay + vy * t
            d = dist(op.x, op.y, px, py)
        if d < best:
            best = d
    return best


def _best_shot_aim(
    context: Context, ball,
) -> tuple[tuple[float, float] | None, float]:
    """Return the most feasible goal-mouth aim point and its score in [0, 1]."""
    goal_x = context.field.length / 2.0
    half = context.field.goal_width / 2.0 * SHOT_AIM_WIDTH_FRACTION
    n = max(1, SHOT_AIM_POINTS)
    best_aim: tuple[float, float] | None = None
    best_score = 0.0
    for i in range(n):
        frac = 0.0 if n == 1 else -1.0 + 2.0 * i / (n - 1)
        y = frac * half
        clear = _segment_clearance(context, ball.x, ball.y, goal_x, y)
        cf = 1.0 if not math.isfinite(clear) else clamp(clear / SHOT_CORRIDOR_M, 0.0, 1.0)
        d = dist(ball.x, ball.y, goal_x, y)
        df = clamp(1.0 - d / SHOT_MAX_DIST_M, 0.0, 1.0)
        score = cf * df
        if score > best_score:
            best_score = score
            best_aim = (goal_x + GOAL_TARGET_DEPTH_M, y)
    return best_aim, best_score


def _shot_goal_probability(context: Context, ball) -> tuple[
    tuple[float, float] | None, float
]:
    """Estimate the goal probability of the best straight shot.

    Combines the geometric feasibility of each goal-mouth aim point with the
    shot distance, yielding a value in [0, 1] interpreted as goal probability.
    Returns ``(aim, probability)``; ``aim`` is None when no shot line exists.
    """
    return _best_shot_aim(context, ball)


def _notify_pass_receiver(context: Context, players: list[Player], target_id: int) -> None:
    """Tell the intended receiver of a pass to face the ball (global rule).

    Every pass in the strategy funnels through here so the receiving player is
    always turned toward the ball and ready to trap it.
    """
    for p in players:
        if p.id == target_id:
            p.notify_incoming_pass()
            return


def _global_shot_rule(
    context: Context, players: list[Player], store,
) -> Player | None:
    """Poll shot feasibility every ``SHOT_CHECK_INTERVAL_TICKS`` frames.

    Return the shooter to command this frame (and order the shot) when the
    best goal probability exceeds ``SHOT_GOAL_PROB_THRESHOLD``; otherwise
    return None so normal role logic runs.
    """
    store.shot_tick = (getattr(store, "shot_tick", 0) + 1) % max(
        1, SHOT_CHECK_INTERVAL_TICKS
    )
    ball = context.ball
    if ball is None or not players:
        return None
    if store.shot_tick != 0:
        return None

    aim, prob = _shot_goal_probability(context, ball)
    store.shot_last_prob = prob
    if aim is None or prob <= SHOT_GOAL_PROB_THRESHOLD:
        return None

    holder = _select_kickoff_holder(context, players, store)
    if holder is None or holder.pose is None:
        return None
    # Do not shoot from deep in our own third: aim would cross our own goal area.
    if ball.x < KICKOFF_OWN_GOAL_GUARD_X and holder.pose.x < 0:
        return None

    # Every global shot uses extra power on top of the distance-based estimate.
    power = clamp(
        holder.shot_power(aim) + GLOBAL_SHOT_POWER_BONUS,
        KICK_POWER_MIN,
        KICK_POWER_MAX,
    )
    holder.action = "shoot:global"
    holder.strike_toward(aim, power)
    store.shot_taken_at = context.now
    return holder


def _defenders_near_segment(
    context: Context, ax: float, ay: float, bx: float, by: float,
) -> int:
    """Count opponents inside ``KICKOFF_PASS_DEFENDER_RANGE_M`` of segment a→b."""
    vx, vy = bx - ax, by - ay
    length2 = vx * vx + vy * vy
    count = 0
    for r in context.opponents.values():
        op = r.pose
        if op is None:
            continue
        if length2 < 1e-6:
            d = dist(op.x, op.y, ax, ay)
        else:
            t = clamp(((op.x - ax) * vx + (op.y - ay) * vy) / length2, 0.0, 1.0)
            d = dist(op.x, op.y, ax + vx * t, ay + vy * t)
        if d <= KICKOFF_PASS_DEFENDER_RANGE_M:
            count += 1
    return count


def _pass_feasibility(
    context: Context, ball, target: tuple[float, float],
) -> float:
    """Score the probability of a clean pass from ``ball`` to ``target``.

    Combines the straight-line distance with the number of defenders whose
    influence range touches the pass line, so a nearby defender lowers the
    score even when the line is technically clear.
    """
    d = dist(ball.x, ball.y, target[0], target[1])
    if d < PASS_MIN_DIST_M or d > PASS_MAX_DIST_M:
        return 0.0
    clear = _segment_clearance(context, ball.x, ball.y, target[0], target[1])
    cf = 1.0 if not math.isfinite(clear) else clamp(clear / PASS_CORRIDOR_M, 0.0, 1.0)
    df = clamp(1.0 - d / PASS_MAX_DIST_M, 0.0, 1.0)
    # Each defender within its influence range removes a fixed share.
    threats = _defenders_near_segment(context, ball.x, ball.y, target[0], target[1])
    ef = 1.0 / (1.0 + 0.5 * threats)
    return cf * df * ef


def _best_pass_teammate(
    context: Context, holder: Player, players: list[Player],
    allow: set[int] | None = None,
) -> Player | None:
    """Return the most feasible pass receiver above the confidence threshold.

    ``allow`` restricts the candidates by player id. A target that sits deeper
    than ``KICKOFF_OWN_GOAL_GUARD_X`` while the ball is already in our danger
    zone is rejected so a soft, mis-aimed pass cannot become an own goal.
    """
    ball = context.ball
    if ball is None:
        return None
    ball_in_danger = ball.x < KICKOFF_OWN_GOAL_GUARD_X
    best: Player | None = None
    best_score = KICKOFF_PASS_FEASIBILITY_THRESHOLD
    for t in players:
        if t is holder:
            continue
        if allow is not None and t.id not in allow:
            continue
        pose = t.pose
        if pose is None:
            continue
        if ball_in_danger and pose.x < KICKOFF_OWN_GOAL_GUARD_X:
            continue
        score = _pass_feasibility(context, ball, (pose.x, pose.y))
        if score > best_score:
            best_score, best = score, t
    return best


def _kickoff_holder_decide(
    context: Context, holder: Player, players: list[Player], store,
) -> None:
    """Shoot if feasible, else pass to the more feasible teammate, else dribble.

    Every decision frame considers the shot first, as required by the route.
    The pass targets are restricted to the role pair (receiver and auxiliary)
    so the holder does not ping the ball sideways into our own goal.
    """
    ball = context.ball
    if ball is None:
        holder.stop()
        return
    aim, score = _best_shot_aim(context, ball)
    if aim is not None and score >= KICKOFF_SHOT_FEASIBILITY_THRESHOLD:
        holder.action = "kickoff:shoot"
        holder.strike_toward(aim, KICK_POWER_SHOT)
        return
    allow = {
        pid for pid in (getattr(store, "kickoff_receiver", None),
                        getattr(store, "kickoff_aux", None))
        if pid is not None
    }
    mate = _best_pass_teammate(context, holder, players, allow or None)
    if mate is not None and mate.pose is not None:
        pose = mate.pose
        d = dist(ball.x, ball.y, pose.x, pose.y)
        power = min(KICK_POWER_PASS_MAX, KICK_POWER_PASS_BASE + KICK_POWER_PASS_PER_M * d)
        # Global rule: every pass tells the receiver to face the ball.
        _notify_pass_receiver(context, players, mate.id)
        holder.action = f"kickoff:pass{mate.id}"
        holder.strike_toward((pose.x, pose.y), power)
        return
    holder.action = "kickoff:drive"
    holder.strike_toward(opponent_goal(context), KICK_POWER_DEFAULT)


def _kickoff_receiver_decide(
    context: Context,
    receiver: Player,
    players: list[Player],
    store,
    kicker: Player | None,
    aux: Player | None,
) -> None:
    """The receiver passes to the kicker or the auxiliary, whichever is safer.

    Feasibility combines the pass distance with the defender influence range.
    When the two options differ by less than ``KICKOFF_PASS_PREFER_MARGIN``,
    the auxiliary is preferred because it is the more advanced outlet.
    """
    ball = context.ball
    if ball is None or receiver.pose is None:
        receiver.stop()
        return

    options: list[tuple[Player, float]] = []
    for mate in (kicker, aux):
        if mate is None or mate is receiver or mate.pose is None:
            continue
        mp = mate.pose
        options.append((mate, _pass_feasibility(context, ball, (mp.x, mp.y))))
    if not options:
        receiver.action = "kickoff:drive"
        receiver.strike_toward(opponent_goal(context), KICK_POWER_DEFAULT)
        return

    best_mate, best_score = max(options, key=lambda item: item[1])
    if aux is not None:
        aux_opt = next((item for item in options if item[0] is aux), None)
        if (
            aux_opt is not None
            and aux_opt[1] >= best_score - KICKOFF_PASS_PREFER_MARGIN
            and best_score > KICKOFF_PASS_FEASIBILITY_THRESHOLD
        ):
            best_mate, best_score = aux_opt

    if best_score <= KICKOFF_PASS_FEASIBILITY_THRESHOLD:
        receiver.action = "kickoff:drive"
        receiver.strike_toward(opponent_goal(context), KICK_POWER_DEFAULT)
        return

    pose = best_mate.pose
    d = dist(ball.x, ball.y, pose.x, pose.y)
    power = min(KICK_POWER_PASS_MAX, KICK_POWER_PASS_BASE + KICK_POWER_PASS_PER_M * d)
    _notify_pass_receiver(context, players, best_mate.id)
    receiver.action = f"kickoff:pass{best_mate.id}"
    receiver.strike_toward((pose.x, pose.y), power)


def _select_kickoff_holder(
    context: Context, players: list[Player], store,
) -> Player:
    """Pick the ball holder with hysteresis to prevent role flapping."""
    ranked = sorted(players, key=lambda p: _player_dist_to_ball(context, p))
    best = ranked[0]
    best_d = _player_dist_to_ball(context, best)
    cur_id = getattr(store, "kickoff_holder", None)
    if cur_id is not None:
        cur = next((p for p in players if p.id == cur_id), None)
        if cur is not None:
            cur_d = _player_dist_to_ball(context, cur)
            if cur_d <= best_d + ATTACKER_KEEP_DIST_MARGIN_M:
                return cur
    store.kickoff_holder = best.id
    return best


def _init_kickoff_route(context: Context, players: list[Player], store) -> None:
    """Assign kicker/receiver/auxiliary roles and reset the route memory."""
    taker = _select_closest_attacker(context, players)
    store.kickoff_taker = taker.id

    rest = [p for p in players if p is not taker]
    receiver = None
    aux = None
    if rest:
        receiver = min(rest, key=lambda p: _player_dist_to_ball(context, p))
        rest = [p for p in rest if p is not receiver]
    if rest:
        aux = min(
            rest,
            key=lambda p: _player_dist_to_position(
                p, (context.field.length / 2.0, 0.0),
            ),
        )
    store.kickoff_receiver = receiver.id if receiver is not None else None
    store.kickoff_aux = aux.id if aux is not None else None
    store.kickoff_route_active = True
    store.kickoff_started_at = context.now
    store.kickoff_ball_played = False
    store.kickoff_advanced = False
    store.kickoff_aux_at_slot = False
    store.kickoff_holder = None
    store.kickoff_lineup_ready = False


def _run_kickoff_route(context: Context, players: list[Player], store) -> None:
    """Run the two-snapshot kickoff route.

    Snapshot 1 (the kick): the kicker stands on the ball-to-receiver line and
    passes straight into the receiver's feet; the auxiliary screens wide.

    Snapshot 2 (first tactical point): the kicker runs to slot 1 while bending
    its path slightly toward the nearest defender (挡拆), the receiver dribbles
    the ball to slot 2, and the auxiliary runs to slot 3 keeping clear of
    defenders. From there the ball holder shoots when the shot is feasible,
    otherwise passes to the more feasible teammate.
    """
    ball = context.ball
    if ball is None or not players:
        return
    if store.kickoff_started_at is None:
        store.kickoff_started_at = context.now

    slot1, slot2, slot3 = _kickoff_slots(context)
    by_id = {p.id: p for p in players}
    kicker = by_id.get(store.kickoff_taker)
    receiver = by_id.get(store.kickoff_receiver)
    aux = by_id.get(store.kickoff_aux)

    if dist(ball.x, ball.y, 0.0, 0.0) > KICKOFF_BALL_MOVED_M:
        store.kickoff_ball_played = True

    holder = _select_kickoff_holder(context, players, store)
    holder_d = _player_dist_to_ball(context, holder)
    opp_d = _nearest_opponent_dist_to_point(context, (ball.x, ball.y))
    ours_holds = opp_d is None or holder_d <= opp_d + KICKOFF_POSSESSION_MARGIN_M
    holder_active = store.kickoff_ball_played and ours_holds

    return _run_kickoff_before_kick(
        context, players, store, kicker, receiver, aux, slot1, slot2, slot3,
        holder, holder_active,
    )


def _on_spot(pose, spot: tuple[float, float]) -> bool:
    """Return whether ``pose`` is within the line-up tolerance of ``spot``."""
    if pose is None:
        return False
    return dist(pose.x, pose.y, spot[0], spot[1]) <= KICKOFF_READY_DIST_M


def _run_kickoff_before_kick(
    context: Context,
    players: list[Player],
    store,
    kicker: Player | None,
    receiver: Player | None,
    aux: Player | None,
    slot1: tuple[float, float],
    slot2: tuple[float, float],
    slot3: tuple[float, float],
    holder: Player,
    holder_active: bool,
) -> None:
    """Run one route frame, waiting for the line-up before the kick.

    Until the ball has been played, the receiver and the screening player walk
    to their kickoff standing spots and the kicker holds the ball at the spot.
    The kick is only taken once both of them are on-spot (or the ready timeout
    expires), so the pass always has a stationary target.
    """
    receiver_spot = KICKOFF_RECEIVE_SPOT
    aux_spot = KICKOFF_AUX_START
    ball = context.ball

    if not store.kickoff_ball_played:
        if store.kickoff_started_at is None:
            store.kickoff_started_at = context.now
        ready = (
            _on_spot(receiver.pose if receiver is not None else None, receiver_spot)
            and _on_spot(aux.pose if aux is not None else None, aux_spot)
        )
        if not ready and context.now - store.kickoff_started_at > KICKOFF_READY_TIMEOUT_S:
            ready = True
        store.kickoff_lineup_ready = ready

        if receiver is not None:
            receiver.action = "kickoff:receive"
            receiver.walk_direct(receiver_spot)
        if aux is not None:
            aux.action = "kickoff:screen"
            aux.walk_direct(aux_spot)

        if kicker is not None:
            if ready:
                # Global rule: the kickoff pass notifies the receiver so it
                # faces the ball while the pass travels.
                if receiver is not None:
                    _notify_pass_receiver(context, players, receiver.id)
                kicker.action = "kickoff:pass"
                kicker.kickoff_pass_to(receiver_spot)
            else:
                # Hold a staging spot behind the ball on the pass line instead of
                # idling on the ball: standing still within 3 m of the ball is a
                # penalty, and the stage leaves the kicker pre-aligned.
                kicker.action = "kickoff:wait"
                bx, by = ball.x, ball.y
                d = dist(bx, by, receiver_spot[0], receiver_spot[1])
                if d < 1e-6:
                    kicker.walk_direct((bx, by))
                else:
                    ux = (receiver_spot[0] - bx) / d
                    uy = (receiver_spot[1] - by) / d
                    score_spot = (bx - ux * KICKOFF_PASS_STAGE_M, by - uy * KICKOFF_PASS_STAGE_M)
                    kicker.walk_direct(
                        score_spot,
                        face=angle_to(bx, by, *receiver_spot),
                    )
        return

    # ---- Ball is in play: run the second snapshot. ----
    if kicker is not None:
        if kicker is holder and holder_active:
            _kickoff_holder_decide(context, kicker, players, store)
        else:
            kicker.action = "kickoff:slot1"
            kicker.walk_direct(_screen_target(context, kicker.pose, slot1))

    if receiver is not None:
        if receiver is holder and holder_active:
            # The receiver compares the kicker and the auxiliary and passes to
            # whichever is more feasible, preferring the auxiliary on ties.
            _kickoff_receiver_decide(
                context, receiver, players, store, kicker, aux,
            )
        else:
            receiver.action = "kickoff:slot2"
            receiver.walk_direct(slot2)

    if aux is not None:
        if aux is holder and holder_active:
            _kickoff_holder_decide(context, aux, players, store)
        else:
            # The auxiliary no longer pushes forward: it holds its line-up spot,
            # facing the receiver so it is squared up to receive a pass.
            pose = aux.pose
            if pose is None:
                aux.stop()
            else:
                face = None
                if receiver is not None and receiver.pose is not None:
                    rp = receiver.pose
                    face = angle_to(pose.x, pose.y, rp.x, rp.y)
                store.kickoff_aux_at_slot = True
                aux.action = "kickoff:hold3"
                aux.walk_to((pose.x, pose.y), face=face)


def _continue_kickoff_route(
    context: Context, players: list[Player], store,
) -> bool:
    """Continue the scripted kickoff route into open play, or end it.

    Return True while the route still owns the team. It ends on timeout, when
    the ball reaches the attack third, or when the opponent clearly wins the
    ball.
    """
    if not getattr(store, "kickoff_route_active", False):
        return False
    ball = context.ball
    if ball is None or not players:
        return False
    if store.kickoff_started_at is not None:
        if context.now - store.kickoff_started_at > KICKOFF_ROUTE_MAX_S:
            return False
    if ball.x > KICKOFF_ROUTE_END_X:
        return False

    holder = _select_kickoff_holder(context, players, store)
    holder_d = _player_dist_to_ball(context, holder)
    opp_d = _nearest_opponent_dist_to_point(context, (ball.x, ball.y))
    if opp_d is not None and opp_d + KICKOFF_POSSESSION_MARGIN_M < holder_d:
        return False

    _run_kickoff_route(context, players, store)
    return True


def _act_our_kickoff(context: Context, players: list[Player], store) -> None:
    """Start and run the scripted backward-kick kickoff route.

    The nearest player is the kicker, the next nearest takes the receiver slot,
    and the third becomes the auxiliary forward. Roles persist across the
    countdown and into open play via ``_continue_kickoff_route``.
    """
    if not players:
        return

    active_ids = {p.id for p in players}
    new_kickoff = (
        store.prev_phase != Phase.OUR_KICKOFF
        or store.kickoff_taker not in active_ids
        or not store.kickoff_route_active
    )
    if new_kickoff:
        _init_kickoff_route(context, players, store)

    _run_kickoff_route(context, players, store)


def _act_opp_kickoff(context: Context, players: list[Player]) -> None:
    """Guard with one player while the others wait outside the center circle."""
    if not players:
        return
    gx, gy = own_goal(context)
    guard = min(players, key=lambda p: _player_dist_to_position(p, (gx, gy)))
    guard.guard()

    rest = [p for p in players if p is not guard]
    r = context.field.circle_radius
    slots = [(-r - 0.5, 0.0), (-r - 2.0, 0.5)]
    for p, target in zip(rest, slots):
        p.action = "opp_kickoff:ready"
        p.walk_to(target, avoid_ball=True, avoid_robots=True)


def _setplay_kick_dir(context: Context) -> float:
    """Return the kick direction: from the ball toward the opponent goal."""
    ball = context.ball
    gx, gy = opponent_goal(context)
    if ball is None:
        return angle_to(0.0, 0.0, gx, gy)
    return angle_to(ball.x, ball.y, gx, gy)


def _setplay_receiver_target(
    context: Context, side: float,
) -> tuple[float, float]:
    """Pick an attacking receiver spot in the opponent half.

    Staying inside ``SETPLAY_ATTACK_MIN_X`` at least and the band
    ``|y| <= SETPLAY_ATTACK_HALF_WIDTH_M``, the spot maximises closeness to the
    opponent goal while staying away from defenders. ``side`` (+1/-1) splits
    the two receivers so they never share a lane.
    """
    f = context.field
    max_x = f.length / 2.0 - SETPLAY_RECEIVER_MARGIN_M
    base_y = clamp(
        side * SETPLAY_RECEIVER_SPREAD_Y,
        -SETPLAY_ATTACK_HALF_WIDTH_M,
        SETPLAY_ATTACK_HALF_WIDTH_M,
    )

    best = (max(SETPLAY_ATTACK_MIN_X + 0.1, 0.0), base_y)
    best_score = -math.inf
    steps = 6
    for i in range(steps):
        x = SETPLAY_ATTACK_MIN_X + (max_x - SETPLAY_ATTACK_MIN_X) * i / (steps - 1)
        for j in (-2, -1, 0, 1, 2):
            y = clamp(
                base_y + j * (SETPLAY_RECEIVER_SPREAD_Y / 2.0),
                -SETPLAY_ATTACK_HALF_WIDTH_M,
                SETPLAY_ATTACK_HALF_WIDTH_M,
            )
            d_def = _nearest_opponent_dist_to_point(context, (x, y))
            clearance = 3.0 if d_def is None else min(d_def, 3.0)
            # Closer to the goal is better, plus a bonus for staying clear.
            score = x + SETPLAY_DEFENDER_WEIGHT * clearance
            if score > best_score:
                best_score, best = score, (x, y)
    return best


def _setplay_request_ticks(
    context: Context, store, receiver: Player, target: tuple[float, float],
) -> int:
    """Advance the consecutive-feasible-pass counter for one receiver.

    The receiver asks for the ball once its pass line to the kick point has
    been good for ``SETPLAY_PASS_CHECK_TICKS`` consecutive clock cycles.
    """
    counters = getattr(store, "setplay_ticks", None)
    if counters is None:
        counters = {}
        store.setplay_ticks = counters

    ball = context.ball
    if ball is None:
        counters[receiver.id] = 0
        return 0
    prob = _pass_feasibility(context, ball, target)
    if prob > SETPLAY_PASS_PROB_THRESHOLD:
        counters[receiver.id] = counters.get(receiver.id, 0) + 1
    else:
        counters[receiver.id] = 0
    return counters.get(receiver.id, 0)


def _act_setplay_attack(
    context: Context, players: list[Player], store,
) -> None:
    """Our attacking set play (corner kick / throw-in in the opponent half).

    Both receivers use the same tactic: stay in the opponent half, as close to
    the goal as defenders allow, and face the kick direction while idle. When a
    receiver's pass line has been clean for ``SETPLAY_PASS_CHECK_TICKS``
    consecutive cycles it asks for the ball; the taker passes to the requester
    with the best pass line and that receiver turns toward the kick point.
    """
    ball = context.ball
    if ball is None or not players:
        for p in players:
            p.stop()
        return

    if getattr(store, "setplay_started_at", None) is None:
        store.setplay_started_at = context.now

    kick_dir = _setplay_kick_dir(context)
    taker = _select_closest_attacker(context, players)
    receivers = [p for p in players if p is not taker]

    requesters: list[tuple[Player, int]] = []
    targets: dict[int, tuple[float, float]] = {}
    for i, r in enumerate(receivers):
        side = 1.0 if i == 0 else -1.0
        target = _setplay_receiver_target(context, side)
        targets[r.id] = target
        ticks = _setplay_request_ticks(context, store, r, target)
        if ticks >= SETPLAY_PASS_CHECK_TICKS:
            requesters.append((r, ticks))

    requested_ids = {r.id for r, _ in requesters}

    # Receivers: run to the attacking spot and face the kick direction while
    # idle. A requester additionally faces the ball (the kick point).
    for r in receivers:
        pose = r.pose
        if pose is None:
            r.stop()
            continue
        target = targets[r.id]
        if r.id in requested_ids:
            r.action = "setplay:request"
            r.turn_to_receive()
        elif dist(pose.x, pose.y, target[0], target[1]) <= ARRIVE_DIST * 2.0:
            r.action = "setplay:face"
            r.walk_to((pose.x, pose.y), face=kick_dir)
        else:
            r.action = "setplay:run"
            r.walk_direct(target, face=kick_dir)

    # Taker: pass to the requester with the best pass line.
    if requesters:
        requesters.sort(
            key=lambda item: _pass_feasibility(
                context, ball, (item[0].pose.x, item[0].pose.y)
            ) if item[0].pose is not None else 0.0,
            reverse=True,
        )
        receiver, _ticks = requesters[0]
        rpose = receiver.pose
        if rpose is not None:
            d = dist(ball.x, ball.y, rpose.x, rpose.y)
            power = min(
                KICK_POWER_PASS_MAX,
                KICK_POWER_PASS_BASE + KICK_POWER_PASS_PER_M * d,
            )
            _notify_pass_receiver(context, players, receiver.id)
            taker.action = f"setplay:pass{receiver.id}"
            taker.strike_toward((rpose.x, rpose.y), power)
            store.setplay_ticks = {}
            return

    waited = context.now - store.setplay_started_at
    if waited > SETPLAY_WAIT_TIMEOUT_S and receivers:
        # Fall back to the nearest receiver so the ball is not left waiting.
        receiver = min(
            receivers, key=lambda p: _player_dist_to_ball(context, p),
        )
        rpose = receiver.pose
        if rpose is not None:
            d = dist(ball.x, ball.y, rpose.x, rpose.y)
            power = min(
                KICK_POWER_PASS_MAX,
                KICK_POWER_PASS_BASE + KICK_POWER_PASS_PER_M * d,
            )
            _notify_pass_receiver(context, players, receiver.id)
            taker.action = f"setplay:force{receiver.id}"
            taker.strike_toward((rpose.x, rpose.y), power)
            store.setplay_ticks = {}
            return

    # Nothing requested yet: stage behind the ball on the kick line.
    taker.action = "setplay:wait"
    ax, ay = math.cos(kick_dir), math.sin(kick_dir)
    stage = (ball.x - ax * KICKOFF_PASS_STAGE_M, ball.y - ay * KICKOFF_PASS_STAGE_M)
    taker.walk_direct(stage, face=kick_dir)


def _act_setplay_pass(
    context: Context, players: list[Player], store,
) -> None:
    """Own-half throw-in / goal kick: ordinary pass to the best teammate.

    The taker picks the teammate with the highest pass success probability and
    passes; the receiver is notified so it faces the ball (the kick direction).
    With no viable option it clears the ball upfield.
    """
    ball = context.ball
    if ball is None or not players:
        for p in players:
            p.stop()
        return

    taker = _select_closest_attacker(context, players)
    mate = _best_pass_teammate(context, taker, players)
    if mate is not None and mate.pose is not None:
        pose = mate.pose
        d = dist(ball.x, ball.y, pose.x, pose.y)
        power = min(
            KICK_POWER_PASS_MAX, KICK_POWER_PASS_BASE + KICK_POWER_PASS_PER_M * d,
        )
        _notify_pass_receiver(context, players, mate.id)
        taker.action = f"setplay:pass{mate.id}"
        taker.strike_toward((pose.x, pose.y), power)
        return

    taker.action = "setplay:clear"
    taker.strike_toward(opponent_goal(context), SETPLAY_CLEAR_POWER)


def _act_our_set_play(context: Context, players: list[Player], store) -> None:
    """Dispatch our set play.

    A corner kick and a throw-in in the opponent half share the attacking
    routine; a throw-in in our own half and a goal kick are ordinary passes;
    everything else falls back to normal play.
    """
    set_play = get_set_play_type(context)
    ball = context.ball
    in_opponent_half = ball is not None and ball.x > 0.0

    if set_play == SetPlay.CORNER_KICK:
        _act_setplay_attack(context, players, store)
        return
    if set_play == SetPlay.THROW_IN:
        if in_opponent_half:
            _act_setplay_attack(context, players, store)
        else:
            _act_setplay_pass(context, players, store)
        return
    if set_play == SetPlay.GOAL_KICK:
        _act_setplay_pass(context, players, store)
        return
    _act_normal(context, players, store)


def _act_opp_set_play(context: Context, players: list[Player], store) -> None:
    """Retreat directly from the ball; wait until the official phase changes."""
    ball = context.ball
    # Account for walk_to's arrival tolerance when targeting the 1.5 m boundary.
    radius = OPP_SET_PLAY_CLEARANCE_M + ARRIVE_DIST
    half_l = context.field.length / 2.0 - OPP_SET_PLAY_FIELD_MARGIN_M
    half_w = context.field.width / 2.0 - OPP_SET_PLAY_FIELD_MARGIN_M
    for p in players:
        pose = p.pose
        p.action = "opp_set:wait"
        if ball is None or pose is None:
            p.stop()
            continue
        if dist(pose.x, pose.y, ball.x, ball.y) >= OPP_SET_PLAY_CLEARANCE_M:
            p.stop()
            continue

        # The radial exit is shortest. At the field edge, also consider the
        # retreat circle's intersections with the in-field boundary.
        outward = angle_to(ball.x, ball.y, pose.x, pose.y)
        candidates = [(
            ball.x + math.cos(outward) * radius,
            ball.y + math.sin(outward) * radius,
        )]
        for x in (-half_l, half_l):
            remaining = radius ** 2 - (x - ball.x) ** 2
            if remaining >= 0.0:
                offset = math.sqrt(remaining)
                candidates.extend((x, ball.y + sign * offset) for sign in (-1.0, 1.0))
        for y in (-half_w, half_w):
            remaining = radius ** 2 - (y - ball.y) ** 2
            if remaining >= 0.0:
                offset = math.sqrt(remaining)
                candidates.extend((ball.x + sign * offset, y) for sign in (-1.0, 1.0))
        candidates = [
            t for t in candidates
            if t == clamp_inside_field(context, *t, margin=OPP_SET_PLAY_FIELD_MARGIN_M)
        ]
        if not candidates:
            p.stop()
            continue

        px, py = pose.x, pose.y
        target = min(candidates, key=lambda t: dist(px, py, *t))
        # Short omnidirectional steps follow the retreat line directly.
        # Do not invoke path avoidance: the movement is away from the ball.
        length = dist(pose.x, pose.y, *target)
        ratio = min(1.0, OMNI_DIST / (2.0 * length))
        step = (
            pose.x + (target[0] - pose.x) * ratio,
            pose.y + (target[1] - pose.y) * ratio,
        )
        p.action = "opp_set:retreat"
        p.walk_to(step, face=angle_to(pose.x, pose.y, ball.x, ball.y))


def _act_ready(context: Context, players: list[Player]) -> None:
    """Move each player to a ready position."""
    game = context.game
    our_kickoff = game is not None and game.kicking_team == context.team_id
    field = context.field
    # Our kickoff priority: center-circle edge, goal-area center, then left side.
    if our_kickoff:
        if len(players) >= 1:
            p1 = players[0]
            p1.action = "ready"
            p1.walk_to(
                (-field.circle_radius, 0.0),
                face=0.0,
                avoid_ball=True,
                avoid_robots=True,
            )
        if len(players) >= 2:
            p2 = players[1]
            p2.action = "ready"
            p2.walk_to(
                (-field.length / 2.0 + field.goal_area_length, 0.0),
                face=0.0,
                avoid_ball=True,
                avoid_robots=True,
            )
        if len(players) >= 3:
            p3 = players[2]
            p3.action = "ready"
            p3.walk_to(
                (-0.5, field.circle_radius + 2),
                face=0.0,
                avoid_ball=True,
                avoid_robots=True,
            )
    # Opponent kickoff priority: outside center circle, goal area, then penalty area.
    else:
        if len(players) >= 1:
            p1 = players[0]
            p1.action = "ready"
            p1.walk_to(
                (-field.circle_radius - 0.5, 0.0),
                face=0.0,
                avoid_ball=True,
                avoid_robots=True,
            )
        if len(players) >= 2:
            p2 = players[1]
            p2.action = "ready"
            p2.walk_to(
                (-field.length / 2.0 + field.goal_area_length, 0.0),
                face=0.0,
                avoid_ball=True,
                avoid_robots=True,
            )
        if len(players) >= 3:
            p3 = players[2]
            p3.action = "ready"
            p3.walk_to(
                (-field.length / 2.0 + field.penalty_area_length, 0.0),
                face=0.0,
                avoid_ball=True,
                avoid_robots=True,
            )



# ======================================================================
# Match visualization: ball position and each player's distance to the ball
# ======================================================================

def _draw_teammate_marker(p: Player) -> None:
    """Draw teammates in red, using cubes while kicking and spheres otherwise.

    This runs for every player each frame regardless of phase, penalty, or
    readiness. The label contains the player ID and current high-level action
    (``p.action``), with ``[KICK]`` appended while kicking. Shape also indicates
    whether the player is in the kicking state.
    """
    from .framework import debugdraw

    pose = p.pose
    if pose is None:
        return
    red = (1.0, 0.2, 0.2)
    if p.is_kicking:
        debugdraw.cube(pose.x, pose.y, rgb=red, scale=0.38, ns="teammate")
    else:
        debugdraw.point(pose.x, pose.y, rgb=red, scale=0.3, ns="teammate")
    kick_tag = " [KICK]" if p.is_kicking else ""
    label = f"{p.id}:{p.action}{kick_tag}"
    debugdraw.text(pose.x, pose.y, label, rgb=(1.0, 0.9, 0.6), ns="teammate_id")


def _analyze_and_draw(context: Context, players: list[Player], store) -> None:
    """Calculate player-to-ball distances and draw them each frame.

    This no longer depends on the analysis module; distances use the ball's
    current position.
    """
    from .framework import debugdraw

    ball = context.ball

    # Nothing can be drawn when the ball is not visible.
    if ball is None:
        return

    # 1. Draw the ball's current position as a green point.
    debugdraw.point(ball.x, ball.y, rgb=(0.0, 1.0, 0.0), scale=0.2, ns="ball_current")

    # 2. Draw distance labels: red for teammates and blue for opponents.
    for p in players:
        pose = p.pose
        if pose is None:
            continue
        d = dist(pose.x, pose.y, ball.x, ball.y)
        debugdraw.text(
            pose.x + 0.3, pose.y - 0.3, f"{d:.1f}m",
            rgb=(1.0, 0.6, 0.6), ns="dist_ours",
        )
    for r in context.opponents.values():
        pose = r.pose
        if pose is None:
            continue
        d = dist(pose.x, pose.y, ball.x, ball.y)
        debugdraw.text(
            pose.x + 0.3, pose.y - 0.3, f"{d:.1f}m",
            rgb=(0.6, 0.6, 1.0), ns="dist_opp",
        )

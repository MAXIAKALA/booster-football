"""Central configuration point for strategy tuning.

This module contains tunable strategy, movement, kicking, and obstacle
avoidance parameters. Existing values were moved here from their original
modules; make future parameter adjustments here first.
"""

from __future__ import annotations

import math


# ======================================================================
# Kick power
# ======================================================================

# Player.kick() clamps power to this range. Unsupported values may cause a fall.
KICK_POWER_MIN = 1.0
KICK_POWER_MAX = 10.0

# Temporary switch: when True, play() runs the kick power→distance sweep
# (src/calibration.py) instead of the match strategy and logs CALIB_* results.
# Off for normal play. For clean gradient data, run with no opposing AI.
ENABLE_KICK_CALIBRATION = False

# Temporary switch: when True, play() runs the kickoff power-sweep experiment
# (src/kickoff_experiment.py) instead of the match strategy. The kicker stages
# behind the ball on the opponent side, faces our own half, kicks toward our half
# with the counter-selected power, records v0 / a0 / distance / direction, then
# stops the simulation. Counter starts at 1 and increments on each valid record;
# at 10 valid samples everything terminates. Off for normal play.
ENABLE_KICKOFF_EXPERIMENT = False
KICKOFF_EXP_STAGE_DIST_M = 1.0      # Kicker standing gap behind the ball (m)
# When set to a number, force this exact kick power for every trial (ignores the
# counter-based schedule). Set to None to use the counter=power sweep.
KICKOFF_EXP_FIXED_POWER = 8.0

# Kick power during normal play.
KICK_POWER_DEFAULT = 5.0
KICK_POWER_BACKFIELD = 5.0
KICK_POWER_OUR_KICKOFF = 5.0


# ======================================================================
# Player movement control
# ======================================================================

ARRIVE_DIST = 0.15             # Target arrival threshold (m)
OMNI_DIST = 1.0                # Use omnidirectional control below this distance
TURN_THRESHOLD = 0.5           # Turn in place above this heading error (rad)
MAX_LINEAR = 2.0               # Maximum commanded forward speed (m/s)
MAX_ANGULAR = 2.0              # Maximum commanded angular speed (rad/s)
LINEAR_GAIN = 1.5              # Translation proportional gain
ANGULAR_GAIN = 2.0             # Rotation proportional gain


# ======================================================================
# Player kicking and shot planning
# ======================================================================

KICK_ENTER_M = 2.0             # Enter kicking state below this ball distance
KICK_EXIT_M = 2.5              # Leave kicking state above this ball distance
CHASE_BEHIND_M = 0.35          # Distance behind the ball while chasing


# ======================================================================
# Player skill parameters: guarding and support
# ======================================================================

GUARD_FACE_BALL = True
GUARD_THREAT_ENTER_X = -1.0
GUARD_THREAT_EXIT_X = -0.7

SUPPORT_DIST_M = 3.0

# ======================================================================
# Normal-phase strategy
# ======================================================================

ATTACKER_KEEP_DIST_MARGIN_M = 0.3  # Prevent attacker-selection oscillation

FALLEN_COST = 10.0  # Distance penalty assigned to a fallen player (m)


# ======================================================================
# Kickoff and set-play strategy
# ======================================================================

# Kickoff
KICKOFF_STAGE_M = 2.0
KICKOFF_FRONT_MARGIN = 0.1
KICKOFF_LATERAL_TOL = 0.35

CENTER_LEAVE_DIST_M = 0.15 # Distance from center at which the ball counts as moved

OPP_SET_WALL_DIST_M = 2.0 # Blocking distance during an opponent kickoff


# ----------------------------------------------------------------------
# Kickoff route: backward pass, role slots, and follow-up decisions
# ----------------------------------------------------------------------
# All coordinates are team-relative (+x points at the opponent goal) and come
# from the two reference snapshots: the kickoff line-up and the first tactical
# point after the kick.

# Backward kick from the kickoff spot. The kicker aims straight at this spot so
# the ball travels along one line into the receiver's feet. Power stays low: the
# simulator's kick direction error grows with power, so a hard pass drifts out.
KICKOFF_RECEIVE_SPOT = (-2.17, 0.95)   # Receiver's kickoff standing spot
# The simulator's kick is far stronger and less accurate than the power number
# suggests: power 3.0 sent a 2.4 m pass 4.7 m and into our own goal. Keep the
# kickoff pass very soft so it cannot overshoot past the receiver.
KICK_POWER_KICKOFF_BACK = 1.2          # Power for the backward kickoff pass

# Kickoff-phase line-up (图1). The pass is only taken once the receiver and the
# screening player have actually reached these spots.
KICKOFF_AUX_START = (-1.78, -2.13)     # 掩护球员 (aux) kickoff standing spot
KICKOFF_READY_DIST_M = 0.6             # Count a player as on-spot below this (m)
# Keep the wait short: the kicker holds a staging spot near the ball, and an
# idle robot within 3 m of the ball is penalised after 10 s.
KICKOFF_READY_TIMEOUT_S = 6.0          # Take the kick anyway after this (s)
KICKOFF_PASS_STAGE_M = 1.2             # Kicker staging gap behind the ball (m)

# Post-kickoff tactical slots (图2 的第一个战术点).
KICKOFF_SLOT_KICKER = (0.93, 1.81)     # 坐标1: kicker after the kick
KICKOFF_SLOT_RECEIVER = (-1.40, 0.26)  # 坐标2: receiver dribbles to here
KICKOFF_SLOT_AUX = (1.24, -1.96)       # 坐标3: auxiliary 下顺 target
KICKOFF_FIELD_MARGIN_M = 0.5           # Keep slots this far inside the touchline

# Kicker screening: bend the path slightly toward the nearest defender.
KICKOFF_KICKER_SCREEN_M = 0.6
# Auxiliary keeps this clearance from defenders while moving to its slot.
KICKOFF_AUX_AVOID_M = 1.2
KICKOFF_AUX_SIDESTEP_M = 0.9
# Gentle touches while dribbling the ball to a slot.
KICK_POWER_DRIBBLE = 1.2

# Every pass notifies the receiver, which then keeps its body facing the ball.
PASS_FACE_WINDOW_S = 4.0

# Global shot rule: consume one tick per frame, evaluate shot feasibility every
# SHOT_CHECK_INTERVAL_TICKS frames and shoot when the goal probability is high.
SHOT_CHECK_INTERVAL_TICKS = 10
SHOT_GOAL_PROB_THRESHOLD = 0.70
# Kick calibration: roughly this much ball travel per unit of kick power (m).
KICK_DIST_PER_POWER = 0.53

# Shot / pass decisions. The ball counts as in play once it leaves the spot.
KICKOFF_BALL_MOVED_M = 0.5         # Ball center must clear this from (0, 0)
KICKOFF_SHOT_FEASIBILITY_THRESHOLD = 0.55
KICKOFF_PASS_FEASIBILITY_THRESHOLD = 0.35

# Global shot rule: add this to the computed power for every global shot.
GLOBAL_SHOT_POWER_BONUS = 2.0

# Kickoff passing: distance plus a defender-influence radius, and the score
# margin within which the auxiliary receiver is preferred over the kicker.
KICKOFF_PASS_DEFENDER_RANGE_M = 2.0
KICKOFF_PASS_PREFER_MARGIN = 0.10

SHOT_AIM_POINTS = 5                # Sample points across the goal mouth
SHOT_AIM_WIDTH_FRACTION = 0.8      # Use this fraction of the goal width
SHOT_CORRIDOR_M = 0.9              # Clear corridor for a full-confidence shot
SHOT_MAX_DIST_M = 10.0             # Shots longer than this lose confidence

PASS_MIN_DIST_M = 1.0
PASS_MAX_DIST_M = 9.0
PASS_CORRIDOR_M = 0.8              # Clear corridor for a full-confidence pass
KICK_POWER_PASS_BASE = 1.2
KICK_POWER_PASS_PER_M = 0.25
KICK_POWER_PASS_MAX = 2.5
KICK_POWER_SHOT = 6.0

# Auxiliary reaction when it reaches slot 3.
KICKOFF_AUX_NEAR_GOAL_INSET_X = 1.5    # Move this far in from the goal line
KICKOFF_AUX_NEAR_GOAL_INSET_Y = 1.5
KICKOFF_AUX_HYSTERESIS_M = 0.3

# Route lifetime and exit conditions. The window must cover the pass, the
# dribble to slot 2, and one shot/pass decision.
KICKOFF_ROUTE_MAX_S = 14.0         # Abandon the route after this long (s)
KICKOFF_ROUTE_END_X = 2.5          # Hand back to normal play past this x (m)
KICKOFF_POSSESSION_MARGIN_M = 0.4  # Opponent this much closer means no ball
# Never kick toward a target deeper than this x while the ball is in our own
# danger zone; a mis-aimed pass there becomes an own goal.
KICKOFF_OWN_GOAL_GUARD_X = -3.0

# ----------------------------------------------------------------------
# Our set plays: corner == opponent-half throw-in (same attacking routine)
# ----------------------------------------------------------------------
# Both receivers use one tactic: stay in the opponent half with x above
# SETPLAY_ATTACK_MIN_X and |y| within SETPLAY_ATTACK_HALF_WIDTH_M, hug the goal
# as far as defenders allow, and face the kick direction while idle.
SETPLAY_ATTACK_MIN_X = 3.0
SETPLAY_ATTACK_HALF_WIDTH_M = 3.0
SETPLAY_RECEIVER_MARGIN_M = 1.0     # Keep this far inside the goal line (m)
SETPLAY_RECEIVER_SPREAD_Y = 1.6     # Lateral split between the two receivers
SETPLAY_DEFENDER_WEIGHT = 1.4       # Weight of staying clear of defenders

# A receiver asks for the ball after this many consecutive feasible cycles.
SETPLAY_PASS_CHECK_TICKS = 30
SETPLAY_PASS_PROB_THRESHOLD = 0.70

# Take the set play anyway after this long to avoid an inactivity penalty.
SETPLAY_WAIT_TIMEOUT_S = 6.0
SETPLAY_CLEAR_POWER = 3.5           # Power when no pass option exists

# ----------------------------------------------------------------------
# Defense: high press, with a goalkeeper only when our box is threatened
# ----------------------------------------------------------------------
PRESS_DIST_M = 1.4                  # Press distance from the ball (m)
PRESS_WIDE_M = 2.0                  # Lateral split of the two pressers
DEFENSE_GOALIE_LINE_INSET_M = 0.35   # Keeper's stance in front of the goal line
DEFENSE_GOALIE_GOAL_MARGIN_M = 0.25  # Keeper stays this far inside the posts

# ----------------------------------------------------------------------
# Attack: receiving favour and its pass trigger
# ----------------------------------------------------------------------
# Every ATTACK_EVAL_TICKS cycles the holder compares its own shot success with
# the best teammate's receiving favour and passes when the shot is poor.
ATTACK_EVAL_TICKS = 30
ATTACK_SHOT_PROB_THRESHOLD = 0.60   # Pass only when the shot is below this
ATTACK_RECEIVE_PROB_THRESHOLD = 0.60  # ...and a teammate scores above this
# Weights of the three receiving-favour dimensions (they sum to 1.0).
RECEIVE_W_GOAL = 0.45               # Closeness to the opponent goal
RECEIVE_W_SPACE = 0.30              # Room around the receiver
RECEIVE_W_LINE = 0.25               # Ball near the receiver-to-goal line

# Opponent set plays: retreat threshold and field boundary margin (m).
OPP_SET_PLAY_CLEARANCE_M = 1.5
OPP_SET_PLAY_FIELD_MARGIN_M = 0.3


# ======================================================================
# Positioning and avoidance
# ======================================================================

OPPONENT_RESTART_AVOID_M = 1.65
CIRCLE_MARGIN_M = 0.3


# ======================================================================
# Kick-target geometry
# ======================================================================

GOAL_TARGET_DEPTH_M = 0.25              # Kick target depth inside the goal (m)

# ======================================================================
# Obstacle geometry
# ======================================================================

BALL_OBSTACLE_RADIUS = 0.5              # Ball obstacle radius (m)
OPPONENT_RADIUS = 0.55                  # Opponent robot radius (m)
TEAMMATE_RADIUS = 0.48                  # Teammate robot radius (m)
SAFETY_MARGIN = 0.22                    # General safety margin (m)

GOAL_DEPTH = 0.6                        # Goal depth used for obstacle modeling (m)
POST_RADIUS = 0.18                      # Goal-post radius (m)
NET_RADIUS = 0.20                       # Goal-net sample radius (m)
NET_STEP = 0.35                         # Goal-net sampling interval (m)

START_IGNORE = 0.0                      # Ignore obstacles near the start (m)
TARGET_IGNORE = 0.0                     # Ignore obstacles near the target (m)

# ======================================================================
# Global path planner (A* grid planner)
# ======================================================================

USE_GLOBAL_PATH_PLANNER = True          # Fall back to local VFH when disabled
GLOBAL_GRID_RESOLUTION_M = 0.35         # Grid cell size; smaller is finer (m)
GLOBAL_FIELD_MARGIN_M = 0.25            # Extra field margin for boundary paths (m)
GLOBAL_OBSTACLE_MARGIN_M = 0.10         # Additional obstacle inflation radius (m)
GLOBAL_PATH_LOOKAHEAD_M = 0.9           # Lookahead distance along the path (m)

# ======================================================================
# Local path planner (VFH direction scan)
# ======================================================================

PLAN_LOOKAHEAD = 1.2                    # Forward probe length (m)
PLAN_CLEARANCE = 0.35                   # Minimum clearance for a candidate (m)
PLAN_STEP = math.radians(15)            # Candidate direction scan step (rad)
PLAN_MAX_OFFSET = math.radians(100)     # Maximum offset from target heading (rad)

# ======================================================================
# Visualization
# ======================================================================

KICK_TARGET_MARK_SIZE_M = 0.18

"""Host-runnable tests for ``src.calibration`` (compact suite).

Run: ``python3 -m unittest tests.test_calibration -v``
"""

from __future__ import annotations

import math
import os
import sys
import types
import unittest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src import calibration as cal  # noqa: E402


class _Ball:
    x = 0.0
    y = 0.0


class _Field:
    length = 14.0
    width = 9.0


class _Ctx:
    def __init__(self, dt=0.033):
        self.now = 0.0
        self.dt = dt
        self.ball = _Ball()
        self.field = _Field()
        self.opponents = {}


class _Pose:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class _Player:
    def __init__(self, pid=1, x=-1.0, y=0.0):
        self.id = pid
        self._pose = _Pose(x, y)
        self.action = ""
        self.kicks = []
        self.stops = 0

    @property
    def pose(self):
        return self._pose

    def walk_to(self, target, **_k):
        self._pose = _Pose(target[0], target[1])
        return True

    def move_to_position(self, target):
        self._pose = _Pose(target[0], target[1])

    def kick(self, d, p):
        self.kicks.append((d, p))

    def release_kick(self):
        return None

    def stop(self):
        self.stops += 1


class _Store(types.SimpleNamespace):
    pass


class PureTests(unittest.TestCase):
    def test_table(self):
        self.assertEqual(list(cal.POWER_TABLE), [float(p) for p in range(1, 11)])
        self.assertEqual(cal.build_power_schedule(), list(cal.POWER_TABLE))

    def test_distance(self):
        self.assertEqual(cal.travel_distance((0, 0), []), 0.0)
        self.assertAlmostEqual(
            cal.travel_distance((0, 0), [(0, 0), (3, 4)]), 5.0)

    def test_launch_fit_constant_speed(self):
        # 0.1 m per 0.05 s -> 2 m/s, zero acceleration.
        s = [(i * 0.05, i * 0.1, 0.0) for i in range(6)]
        v0, a0 = cal.launch_fit(s)
        self.assertAlmostEqual(v0, 2.0, places=2)
        self.assertAlmostEqual(a0, 0.0, places=1)

    def test_launch_fit_accelerating(self):
        # s(t) = 0.5*a*t^2 with a = 10 -> v0 ~ 0, a0 ~ 10.
        s = [(i * 0.05, 5.0 * (i * 0.05) ** 2, 0.0) for i in range(6)]
        _v0, a0 = cal.launch_fit(s)
        self.assertAlmostEqual(a0, 10.0, places=1)

    def test_launch_fit_no_motion(self):
        s = [(i * 0.05, 0.0, 0.0) for i in range(4)]
        self.assertEqual(cal.launch_fit(s), (0.0, 0.0))

    def test_linear_fit(self):
        self.assertAlmostEqual(cal.linear_fit([1, 2, 3], [2, 4, 6])[0], 2.0)
        self.assertEqual(cal.linear_fit([1.0], [2.0]), (0.0, 0.0, 0.0))


class MachineTests(unittest.TestCase):
    def _init(self):
        ctx, store, p = _Ctx(), _Store(), _Player()
        cal.KickCalibrator().update(ctx, [p], store)
        return ctx, store, p

    def test_requires_gap_before_kick(self):
        ctx, store, p = self._init()
        # Player starts 1.0 m behind as fake snaps; first frame should kick.
        self.assertIn(store.cal_state, ("strike", "approach"))

    def test_waits_when_ball_moving(self):
        ctx, store, p = self._init()
        # Force approach and simulate a moving ball -> no kick.
        store.cal_state = "approach"
        p.kicks.clear()
        for i in range(4):
            ctx.now = float(i)
            ctx.ball.x = float(i) * 0.5
            cal.KickCalibrator().update(ctx, [p], store)
        self.assertEqual(len(p.kicks), 0)

    def test_no_kick_when_opponent_close(self):
        ctx, store, p = self._init()
        store.cal_state = "approach"
        p.kicks.clear()
        ctx.ball.x = ctx.ball.y = 0.0
        ctx.now = 5.0
        # Keep ball still long enough, but opponent sits on it.
        store.cal_ball_still_since = 0.0
        opponent = types.SimpleNamespace(pose=_Pose(0.5, 0.0))
        ctx.opponents = {1: opponent}
        cal.KickCalibrator().update(ctx, [p], store)
        self.assertEqual(len(p.kicks), 0)

    def test_record_accepts_moved_ball(self):
        ctx, store, p = self._init()
        # Satisfy the ball-rest gate, then let it kick.
        store.cal_ball_still_since = -10.0
        ctx.now = 0.0
        cal.KickCalibrator().update(ctx, [p], store)
        self.assertEqual(store.cal_state, "strike")
        base = ctx.now
        # Move the ball every frame (contact happens during strike), then track.
        for i in range(1, 40):
            ctx.now = base + 0.05 * i
            ctx.ball.x += 0.06
            cal.KickCalibrator().update(ctx, [p], store)
            if store.cal_levels[0].result is not None:
                break
        self.assertIsNotNone(store.cal_levels[0].result)
        self.assertGreater(store.cal_levels[0].result.distance, 0.0)
        self.assertAlmostEqual(store.cal_levels[0].result.gap, 1.0, delta=0.2)

    def test_missed_kick_retries(self):
        ctx, store, p = self._init()
        store.cal_ball_still_since = -10.0
        ctx.now = 0.0
        cal.KickCalibrator().update(ctx, [p], store)
        self.assertEqual(store.cal_state, "strike")
        base = ctx.now
        # Ball never moves: strike times out, then reject as missed.
        for i in range(1, 120):
            ctx.now = base + 0.1 * i
            cal.KickCalibrator().update(ctx, [p], store)
            if store.cal_levels[0].reasons:
                break
        self.assertIn("ball_did_not_move", store.cal_levels[0].reasons)

    def test_pauses_without_ball(self):
        ctx, store, p = _Ctx(), _Store(), _Player()
        ctx.ball = None
        cal.KickCalibrator().update(ctx, [p], store)
        self.assertIsNone(getattr(store, "cal_state", None))


if __name__ == "__main__":
    unittest.main()
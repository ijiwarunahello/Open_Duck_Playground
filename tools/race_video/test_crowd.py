import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.chdir(REPO)

from lanes import CKPT  # noqa: E402

UNTRAINED = str(CKPT / "2026_11_27_014427_0.onnx")


def test_lane_reset_restores_start_pose():
    from race_sim import Lane
    ln = Lane(UNTRAINED, "x", lane_y=1.5, lane_x=-2.0)
    start = ln.qpos()
    for _ in range(200):
        ln.step((0.1, 0.0, 0.0))
    ln.data.qpos[2] = 0.05
    ln.step((0.0, 0.0, 0.0))
    ln.data.qvel[:] = 1.0
    ln.reset()
    assert ln.qpos()[:2] == pytest.approx([-2.0, 1.5])
    assert np.allclose(ln.qpos(), start)
    assert not ln.data.qvel.any()
    assert not ln.fallen
    assert not ln.inf.last_action.any()


def _q(roll_deg, z=0.15):
    q = np.zeros(7)
    h = math.radians(roll_deg) / 2
    q[2] = z
    q[3:7] = [math.cos(h), math.sin(h), 0, 0]
    return q


def test_tipped():
    from crowd import tipped
    assert not tipped(_q(0))
    assert not tipped(_q(60))
    assert tipped(_q(100))
    assert tipped(_q(0, z=0.05))


def test_driver_ranges_and_determinism():
    from crowd import Driver
    a, b = Driver(seed=7), Driver(seed=7)
    cmds, pushes = [], []
    for k in range(int(40 / 0.002)):
        ca, pa = a.tick(k * 0.002)
        cb, pb = b.tick(k * 0.002)
        assert ca == cb and pa == pb
        cmds.append(ca)
        if pa is not None:
            pushes.append((k * 0.002, pa))
    cmds = np.array(cmds)
    assert cmds[:, 0].min() >= -0.15 and cmds[:, 0].max() <= 0.15
    assert cmds[:, 1].min() >= -0.2 and cmds[:, 1].max() <= 0.2
    assert cmds[:, 2].min() >= -1.0 and cmds[:, 2].max() <= 1.0
    assert len({tuple(c) for c in cmds}) >= 4  # resampled every 10 s over 40 s
    gaps = np.diff([t for t, _ in pushes])
    assert len(pushes) >= 4 and gaps.min() >= 5.0 - 1e-6 and gaps.max() <= 10.0 + 1e-6
    for _, (vx, vy) in pushes:
        assert 0.1 - 1e-9 <= math.hypot(vx, vy) <= 1.0 + 1e-9


def test_crowd_duck_counts_fall_and_respawns():
    from crowd import CrowdDuck
    d = CrowdDuck(UNTRAINED, x=0.8, y=0.0, seed=1)
    d.lane.data.qpos[2] = 0.05
    d.step(0.0)
    assert d.falls == 1
    assert d.lane.qpos()[2] > 0.1  # back at the start pose
    assert not d.lane.fallen

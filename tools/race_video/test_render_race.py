import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_race import project, result_label  # noqa: E402


def test_project_lookat_is_image_center():
    cam = dict(lookat=(0.5, 1.0, 0.1), distance=3.0, azimuth=90.0, elevation=-35.0)
    u, v = project(np.array([0.5, 1.0, 0.1]), cam, width=1920, height=1080, fovy=45.0)
    assert u == pytest.approx(960, abs=1)
    assert v == pytest.approx(540, abs=1)


def test_project_plus_x_is_screen_right_and_up_is_screen_up():
    cam = dict(lookat=(0.0, 0.0, 0.0), distance=3.0, azimuth=90.0, elevation=-35.0)
    u0, v0 = project(np.array([0.0, 0.0, 0.0]), cam, 1920, 1080, 45.0)
    u1, _ = project(np.array([0.5, 0.0, 0.0]), cam, 1920, 1080, 45.0)
    _, v2 = project(np.array([0.0, 0.0, 0.3]), cam, 1920, 1080, 45.0)
    assert u1 > u0
    assert v2 < v0


@pytest.mark.parametrize(
    "fallen,dx,expected",
    [(True, 0.0, "ころんだ"), (True, 1.5, "ころんだ"), (False, 0.06, "うごかない"), (False, 1.3, "ゴール!"), (False, 0.8, "とちゅう")],
)
def test_result_label(fallen, dx, expected):
    assert result_label(fallen, dx, goal_x=1.2) == expected


def _qpos_with_yaw(deg):
    import math
    q = np.zeros(7)
    h = math.radians(deg) / 2
    q[3:7] = [math.cos(h), 0, 0, math.sin(h)]
    return q


def test_heading_cmd_steers_right_when_yawed_left():
    from render_race import heading_cmd
    vx, vy, wz = heading_cmd(_qpos_with_yaw(20.0))
    assert vx == pytest.approx(0.15)
    assert vy == 0.0
    assert wz < 0


def test_heading_cmd_zero_when_straight_and_clamped():
    from render_race import heading_cmd
    assert heading_cmd(_qpos_with_yaw(0.0))[2] == pytest.approx(0.0)
    assert heading_cmd(_qpos_with_yaw(170.0))[2] == pytest.approx(-1.0)


def test_race_cmd_stops_after_passing_goal():
    from render_race import race_cmd
    q = _qpos_with_yaw(0.0)
    q[0] = 1.2 + 0.31
    assert race_cmd(q, goal_x=1.2, lane_y=0.0) == (0.0, 0.0, 0.0)
    q[0] = 1.0
    assert race_cmd(q, goal_x=1.2, lane_y=0.0)[0] == pytest.approx(0.15)


def test_heading_cmd_returns_toward_lane_center():
    from render_race import heading_cmd
    q = _qpos_with_yaw(0.0)
    q[1] = 0.4  # drifted to +y (the duck's left) of its lane centre at y=0
    assert heading_cmd(q, lane_y=0.0)[2] < 0
    q[1] = -0.4
    assert heading_cmd(q, lane_y=0.0)[2] > 0

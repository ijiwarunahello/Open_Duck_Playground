import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "race_video"))
from curve import ckpt_step, minutes_by_step, partial_polyline, to_pixels  # noqa: E402
from learn_texts import T  # noqa: E402
from render_learning import (  # noqa: E402
    SCENES, S5_NORMAL_BEFORE, S5_SLOW, SLOW_SPEED, bar_frac, ghost_qpos, s5_sim_time, scene_at, yaw_quat,
)
from scored import CARD_FULL, card_values, scaled_terms, step_reward  # noqa: E402
from texts import parse_ruby  # noqa: E402

RAW = {"tracking_lin_vel": 0.5, "tracking_ang_vel": 1.0, "torques": 20.0, "action_rate": 0.4,
       "alive": 1.0, "imitation": 2.0, "stand_still": 0.0}


def test_scaled_terms_use_joystick_scales_and_dt():
    s = scaled_terms(RAW)
    assert s["alive"] == pytest.approx(20.0 * 0.02)
    assert s["action_rate"] == pytest.approx(-0.5 * 0.4 * 0.02)


def test_step_reward_is_clipped_at_zero():
    assert step_reward({**RAW, "alive": 0.0, "tracking_lin_vel": 0.0, "tracking_ang_vel": 0.0,
                        "imitation": -50.0}) == 0.0


def test_card_values_group_terms():
    c = card_values(RAW)
    assert c["alive"] == pytest.approx(20.0)
    assert c["follow"] == pytest.approx(2.5 * 0.5 + 6.0 * 1.0)
    assert c["imitate"] == pytest.approx(2.0)
    assert c["jitter"] == pytest.approx(0.5 * 0.4 + 1e-3 * 20.0)  # positive magnitude of the cost
    assert CARD_FULL["follow"] == pytest.approx(8.5)


def test_minutes_from_checkpoint_names():
    names = ["2026_11_27_014427_0.onnx", "2026_11_27_022458_64389120.onnx", "2026_11_27_045005_300482560.onnx"]
    m = minutes_by_step(names)
    assert m[0] == 0
    assert m[64389120] == pytest.approx(40.517, abs=0.01)
    assert m[300482560] == pytest.approx(185.633, abs=0.01)
    assert ckpt_step(names[1]) == 64389120


def test_to_pixels_and_partial_polyline():
    px = to_pixels([(0, 0), (50, 100), (100, 50)], (0, 0, 100, 100), 100, 100)
    assert px == [(0, 100), (50, 0), (100, 50)]
    half = partial_polyline(px, 0.75)
    assert half[-1] == pytest.approx((75, 25))
    assert partial_polyline(px, 1.0) == px
    assert partial_polyline(px, 0.0) == px[:1]


def test_scene_timeline():
    assert scene_at(0.0) == 1 and scene_at(4.0) == 2 and scene_at(29.99) == 3 and scene_at(59.99) == 6
    assert SCENES[-1] == 60.0


def test_s5_sim_time_is_continuous_and_slow_in_the_window():
    slow_len = (S5_SLOW[1] - S5_SLOW[0]) / SLOW_SPEED
    assert s5_sim_time(S5_NORMAL_BEFORE) == pytest.approx(S5_SLOW[0])
    assert s5_sim_time(S5_NORMAL_BEFORE + slow_len) == pytest.approx(S5_SLOW[1])
    a, b = s5_sim_time(S5_NORMAL_BEFORE + 1.0), s5_sim_time(S5_NORMAL_BEFORE + 2.0)
    assert b - a == pytest.approx(SLOW_SPEED)
    # scene 5 must fit: its end maps to a sim time after the slow window
    assert s5_sim_time(SCENES[5] - SCENES[4]) > S5_SLOW[1]


def test_bar_frac_clamps():
    assert bar_frac(-1, 5) == 0.0 and bar_frac(10, 5) == 1.0 and bar_frac(2.5, 5) == 0.5


def test_ghost_is_upright_with_duck_yaw_and_reference_joints():
    yaw = math.radians(30)
    duck = np.zeros(21)
    duck[:3] = [1.0, 0.0, 0.15]
    # yawed and tilted body
    tilt = np.array([math.cos(0.2), math.sin(0.2), 0, 0])
    yq = np.array([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])
    w1, x1, y1, z1 = yq
    w2, x2, y2, z2 = tilt
    duck[3:7] = [w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                 w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2]
    ref = np.arange(16, dtype=float)
    addr = np.arange(7, 21)
    g = ghost_qpos(duck, ref, addr, stand_z=0.16, dy=0.3)
    assert g[1] == pytest.approx(0.3) and g[2] == pytest.approx(0.16)
    assert g[4] == pytest.approx(0.0) and g[5] == pytest.approx(0.0)
    assert np.allclose(g[3:7], yaw_quat(duck))
    assert list(g[7:21]) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15]  # antennas (9, 10) dropped


def _all_texts(d):
    for v in d.values():
        if isinstance(v, str):
            yield v
        elif isinstance(v, dict):
            yield from _all_texts(v)
        else:
            yield from v


def test_every_kanji_has_ruby():
    for s in _all_texts(T):
        for base, ruby in parse_ruby(s):
            if ruby is None:
                assert not any("一" <= ch <= "鿿" for ch in base), s

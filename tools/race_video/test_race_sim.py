import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.chdir(REPO)

from race_sim import Lane, load_lanes  # noqa: E402

CKPT = Path(os.environ.get("RACE_CKPT_DIR", "/home/iri/Workspaces/src/github.com/ijiwarunahello/Open_Duck_Playground/checkpoints/backlash_20261127_0141"))
UNTRAINED = str(CKPT / "2026_11_27_014427_0.onnx")
TRAINED = str(CKPT / "2026_11_27_045005_300482560.onnx")
FORWARD = (0.15, 0.0, 0.0)


def run(lane, seconds, cmd=FORWARD):
    for _ in range(int(seconds / 0.002)):
        lane.step(cmd)


def test_untrained_stays():
    lane = Lane(UNTRAINED, "0", lane_y=0.0)
    x0 = lane.qpos()[0]
    run(lane, 6.0)
    assert abs(lane.qpos()[0] - x0) < 0.05


def test_trained_advances():
    lane = Lane(TRAINED, "3h", lane_y=0.0)
    x0 = lane.qpos()[0]
    run(lane, 6.0)
    assert lane.qpos()[0] - x0 > 0.3


def test_lane_y_offsets_start():
    lane = Lane(UNTRAINED, "0", lane_y=1.0)
    assert lane.qpos()[1] == pytest.approx(1.0)


def test_push_adds_velocity():
    lane = Lane(UNTRAINED, "0", lane_y=0.0)
    before = lane.data.qvel[1]
    lane.push(0.0, 0.5)
    assert lane.data.qvel[1] - before == pytest.approx(0.5)


def test_fallen_latches():
    lane = Lane(UNTRAINED, "0", lane_y=0.0)
    lane.data.qpos[2] = 0.05
    lane.step((0.0, 0.0, 0.0))
    assert lane.fallen
    lane.data.qpos[2] = 0.15
    lane.step((0.0, 0.0, 0.0))
    assert lane.fallen


def test_missing_onnx_fails():
    with pytest.raises(FileNotFoundError):
        load_lanes([(UNTRAINED, "0"), ("/nonexistent/x.onnx", "bad")])


def test_load_lanes_spacing():
    lanes = load_lanes([(UNTRAINED, "a"), (TRAINED, "b")], spacing=0.5)
    assert [ln.label for ln in lanes] == ["a", "b"]
    assert lanes[1].qpos()[1] - lanes[0].qpos()[1] == pytest.approx(0.5)

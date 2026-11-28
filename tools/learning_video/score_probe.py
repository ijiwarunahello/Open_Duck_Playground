"""Accumulate the training reward terms for checkpoints walking forward 8 s.

    python tools/learning_video/score_probe.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "race_video"))
from lanes import CKPT  # noqa: E402
from scored import ScoredLane, scaled_terms, step_reward  # noqa: E402

CKPTS = [("0", "2026_11_27_014427_0"), ("40", "2026_11_27_022458_64389120"),
         ("180", "2026_11_27_045005_300482560")]


def run(onnx, seconds=8.0, cmd=(0.15, 0.0, 0.0)):
    lane = ScoredLane(str(CKPT / f"{onnx}.onnx"), "", 0.0)
    tot = {}
    total = 0.0
    x0 = lane.data.qpos[0]
    for _ in range(int(seconds / 0.002)):
        lane.step(cmd)
        if lane.last is not None and lane._counter % lane.inf.decimation == 0:
            for k, v in scaled_terms(lane.last["raw"]).items():
                tot[k] = tot.get(k, 0.0) + v
            total += step_reward(lane.last["raw"])
    return tot, total, lane.data.qpos[0] - x0, lane.fallen


if __name__ == "__main__":
    for label, onnx in CKPTS:
        tot, total, dx, fallen = run(onnx)
        print(f"{label:>4} min dx={dx:+.2f} fallen={fallen} total={total:.1f} "
              + " ".join(f"{k}={v:+.2f}" for k, v in tot.items()))

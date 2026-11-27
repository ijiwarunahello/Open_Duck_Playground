"""Many ducks practising at once, under the same conditions as training
(playground/open_duck_mini_v2/joystick.py):
- command (vx, vy, wz) resampled every 10 s from [-0.15,0.15] x [-0.2,0.2] x [-1,1]
- a push every 5-10 s, 0.1-1.0 m/s in a random horizontal direction
- an episode ends when the body tips past 90 deg (or the base drops below 0.08 m) and the
  duck starts again from the standing pose.

Each duck is an independent MuJoCo model; ducks are split across processes.
Run with cwd = repository root.
"""
import math
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from race_sim import FALL_Z, Lane

DT = 0.002
CMD_PERIOD_S = 10.0
CMD_RANGE = ((-0.15, 0.15), (-0.2, 0.2), (-1.0, 1.0))
PUSH_INTERVAL_S = (5.0, 10.0)
PUSH_MAG = (0.1, 1.0)
SPACING = 0.8


def tipped(qpos) -> bool:
    """Training's termination (body z axis points below the horizon) or a low base."""
    w, x, y, z = qpos[3:7]
    up_z = 1 - 2 * (x * x + y * y)  # world-z component of the body z axis
    return up_z < 0.0 or qpos[2] < FALL_Z


class Driver:
    """Per-duck random commands and pushes, deterministic for a seed."""

    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)
        self.cmd = self._sample_cmd()
        self.next_cmd_t = CMD_PERIOD_S * self.rng.uniform(0.0, 1.0)  # stagger across ducks
        self.next_push_t = self.rng.uniform(*PUSH_INTERVAL_S)

    def _sample_cmd(self):
        return tuple(float(self.rng.uniform(lo, hi)) for lo, hi in CMD_RANGE)

    def tick(self, t: float):
        """Returns (command, push or None) for sim time t."""
        if t >= self.next_cmd_t:
            self.cmd = self._sample_cmd()
            self.next_cmd_t += CMD_PERIOD_S
        push = None
        if t >= self.next_push_t:
            th = self.rng.uniform(0.0, 2 * math.pi)
            mag = self.rng.uniform(*PUSH_MAG)
            push = (float(mag * math.cos(th)), float(mag * math.sin(th)))
            self.next_push_t += self.rng.uniform(*PUSH_INTERVAL_S)
        return self.cmd, push


class CrowdDuck:
    def __init__(self, onnx_path: str, x: float, y: float, seed: int):
        self.lane = Lane(onnx_path, "", lane_y=y, lane_x=x)
        self.driver = Driver(seed)
        self.falls = 0

    def step(self, t: float) -> None:
        cmd, push = self.driver.tick(t)
        if push is not None:
            self.lane.push(*push)
        self.lane.step(cmd)
        if tipped(self.lane.data.qpos):
            self.falls += 1
            self.lane.reset()


def grid(n_side: int, spacing: float = SPACING):
    c = (n_side - 1) / 2
    return [((i - c) * spacing, (j - c) * spacing) for i in range(n_side) for j in range(n_side)]


def _run_chunk(args):
    onnx_path, cells, seeds, seconds, warmup, fps = args
    import os
    import sys

    sys.stdout = open(os.devnull, "w")  # MjInfer prints a lot per instance
    ducks = [CrowdDuck(onnx_path, x, y, s) for (x, y), s in zip(cells, seeds)]
    n_frames = int(round(seconds * fps)) + 1
    w_steps = round(warmup / DT)
    frames, falls = [], []
    step = 0
    for fi in range(n_frames):
        target = w_steps + round(fi / fps / DT)
        while step < target:
            t = step * DT
            for d in ducks:
                d.step(t)
            step += 1
            if step == w_steps:
                for d in ducks:
                    d.falls = 0  # count falls only inside the recorded window
        frames.append([d.lane.qpos() for d in ducks])
        falls.append(sum(d.falls for d in ducks))
    return np.array(frames), np.array(falls)


def simulate_crowd(onnx_path, n_side, seconds, warmup=0.0, seed=0, fps=30, workers=14):
    """Returns frames (n_frames, n_ducks, nq) and cumulative falls per frame."""
    cells = grid(n_side)
    seeds = [seed * 100003 + i for i in range(len(cells))]
    chunks = [(onnx_path, cells[k::workers], seeds[k::workers], seconds, warmup, fps) for k in range(workers)]
    chunks = [c for c in chunks if c[1]]
    with ProcessPoolExecutor(len(chunks)) as ex:
        parts = list(ex.map(_run_chunk, chunks))
    n = len(cells)
    nf = parts[0][0].shape[0]
    frames = np.zeros((nf, n, parts[0][0].shape[2]))
    falls = np.zeros(nf, dtype=int)
    for k, (fr, fa) in enumerate(parts):
        frames[:, k::len(chunks)] = fr
        falls += fa
    return frames, falls

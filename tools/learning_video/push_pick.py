"""Try the same lateral push at several moments on the 3-hour duck walking forward and
report whether it falls and how far it tilts (used to pick scene 5's push).

    python tools/learning_video/push_pick.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "race_video"))
from lanes import CKPT  # noqa: E402
from race_sim import Lane  # noqa: E402
from render_race import heading_cmd  # noqa: E402

BEST = str(CKPT / "2026_11_27_045005_300482560.onnx")


def tilt_deg(qpos):
    w, x, y, z = qpos[3:7]
    # angle between body z axis and world z
    zz = 1 - 2 * (x * x + y * y)
    return float(np.degrees(np.arccos(np.clip(zz, -1, 1))))


def trial(push_t, vel, seconds=10.0):
    ln = Lane(BEST, "", 0.0)
    max_tilt = 0.0
    for k in range(int(seconds / 0.002)):
        if k == round(push_t / 0.002):
            ln.push(*vel)
        ln.step(heading_cmd(ln.data.qpos, lane_y=0.0))
        if k * 0.002 > push_t:
            max_tilt = max(max_tilt, tilt_deg(ln.data.qpos))
    return ln.fallen, max_tilt, float(ln.data.qpos[0])


if __name__ == "__main__":
    for vy in (0.6, 0.8, 1.0):
        for t in np.arange(3.0, 4.01, 0.25):
            f, mt, x = trial(float(t), (0.0, vy))
            print(f"vy={vy:.1f} t={t:.2f} fallen={f} max_tilt={mt:5.1f} x={x:.2f}", flush=True)

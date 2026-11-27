"""Sweep the sideways push (strength x time) in the same timeline the video uses:
10 s standing, then forward with heading/lane-centre steering, push once for every lane.

Prints which lanes fell for each combination. Which lanes fall at 0.7-0.8 m/s depends on
the gait phase at the moment of the push, so the time matters as much as the strength.
Run with cwd = repository root.
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lanes import GOAL_X, LANES  # noqa: E402
from race_sim import load_lanes  # noqa: E402
import render_race as R  # noqa: E402

MAGNITUDES = [0.4, 0.5, 0.6, 0.7, 0.8]
TIMES = [4.0, 4.25, 4.5, 4.75, 5.0, 5.25, 5.5, 5.75, 6.0]


def trial(vy: float, push_time: float):
    """Returns [(label, fallen, forward distance)] in lane order (0 minutes first)."""
    lanes = load_lanes(LANES, spacing=R.SPACING)
    frames, fallen, _ = R.simulate(lanes, push_time, (0.0, vy), GOAL_X)
    return [(ln.label, bool(fallen[-1, i]), float(frames[-1, i, 0] - frames[0, i, 0])) for i, ln in enumerate(lanes)]


def _row(args):
    vy, t = args
    return vy, t, "".join("F" if f else "." for _, f, _ in trial(vy, t))


if __name__ == "__main__":
    print("fallen lanes, left = 0 minutes ... right = 3 hours")
    with ProcessPoolExecutor(8) as ex:
        for vy, t, row in ex.map(_row, [(vy, t) for vy in MAGNITUDES for t in TIMES]):
            print(f"vy={vy:.2f} t={t:.2f}  {row}", flush=True)

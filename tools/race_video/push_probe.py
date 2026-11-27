"""Find a sideways push that topples weak lanes but not the best one.

Prints, per push magnitude, each lane's result (fallen / forward distance).
Run with cwd = repository root.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lanes import LANES  # noqa: E402
from race_sim import load_lanes  # noqa: E402

PUSH_AT_S = 5.0
RUN_S = 20.0
DT = 0.002
MAGNITUDES = [0.2, 0.4, 0.6, 0.8, 1.0]
FORWARD = (0.15, 0.0, 0.0)


def trial(vy: float):
    lanes = load_lanes(LANES)
    x0 = [ln.qpos()[0] for ln in lanes]
    for i in range(int(RUN_S / DT)):
        if i == int(PUSH_AT_S / DT):
            for ln in lanes:
                ln.push(0.0, vy)
        for ln in lanes:
            ln.step(FORWARD)
    return [(ln.label, ln.fallen, ln.qpos()[0] - x) for ln, x in zip(lanes, x0)]


if __name__ == "__main__":
    labels = [label for _, label in LANES]
    print("push_vy | " + " | ".join(labels))
    for vy in MAGNITUDES:
        cells = [("FALL" if f else f"{dx:+.2f}m") for _, f, dx in trial(vy)]
        print(f"{vy:.1f} | " + " | ".join(cells), flush=True)

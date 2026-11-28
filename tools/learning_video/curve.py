"""Training curve for scene 4: eval/episode_reward of run backlash_20261127_0141,
placed on wall-clock minutes taken from the checkpoint names (YYYY_MM_DD_HHMMSS_step)."""
from datetime import datetime
from pathlib import Path


def ckpt_time(name: str) -> datetime:
    parts = Path(name).stem.split("_")
    return datetime.strptime("_".join(parts[:4]), "%Y_%m_%d_%H%M%S")


def ckpt_step(name: str) -> int:
    return int(Path(name).stem.split("_")[4])


def minutes_by_step(names) -> dict:
    """{step: minutes since the first checkpoint}"""
    t0 = min(ckpt_time(n) for n in names)
    return {ckpt_step(n): (ckpt_time(n) - t0).total_seconds() / 60 for n in names}


def load_curve(run_dir) -> list:
    """[(minutes, episode_reward)] sorted by minutes, one point per evaluated checkpoint."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    run_dir = Path(run_dir)
    names = [p.name for p in run_dir.glob("*.onnx")]
    mins = minutes_by_step(names)
    ea = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    ea.Reload()
    pts = [(mins[e.step], float(e.value)) for e in ea.Scalars("eval/episode_reward") if e.step in mins]
    if len(pts) != len(mins):
        raise ValueError(f"{len(pts)} eval points for {len(mins)} checkpoints")
    return sorted(pts)


def to_pixels(pts, box, x_max, y_max):
    """Data -> pixel coords inside box=(left, top, right, bottom); y grows upward."""
    l, t, r, b = box
    return [(l + (x / x_max) * (r - l), b - (y / y_max) * (b - t)) for x, y in pts]


def partial_polyline(px, frac):
    """The first `frac` (0..1) of the polyline, measured along x, ending on an interpolated point."""
    if frac <= 0:
        return px[:1]
    x_end = px[0][0] + frac * (px[-1][0] - px[0][0])
    out = [px[0]]
    for (x0, y0), (x1, y1) in zip(px, px[1:]):
        if x1 <= x_end:
            out.append((x1, y1))
        else:
            a = (x_end - x0) / (x1 - x0)
            out.append((x_end, y0 + a * (y1 - y0)))
            break
    return out

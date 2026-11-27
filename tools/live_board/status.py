"""Live board status: what a training run has produced so far.

A run directory written by playground/common/runner.py holds one
`<%Y_%m_%d_%H%M%S>_<step>.onnx` per checkpoint (the first at step 0, written
when training starts) and tensorboardX events with `eval/episode_reward`.
"""
import re
import time
from datetime import datetime
from pathlib import Path

CONTROL_DT = 0.02  # one env step = one control period
REWARD_TAG = "eval/episode_reward"
_ONNX = re.compile(r"^(\d{4}_\d{2}_\d{2}_\d{6})_(\d+)\.onnx$")


def practice_days(step: int) -> float:
    """Simulated walking time in days (300M steps -> about 69 days)."""
    return step * CONTROL_DT / 86400


def list_checkpoints(run_dir):
    """-> [(step, unix time, onnx path)] sorted by step."""
    out = []
    for p in Path(run_dir).glob("*.onnx"):
        m = _ONNX.match(p.name)
        if m:
            t = datetime.strptime(m.group(1), "%Y_%m_%d_%H%M%S").timestamp()
            out.append((int(m.group(2)), t, str(p)))
    return sorted(out)


def minutes_since_start(ckpts, i):
    return int(round((ckpts[i][1] - ckpts[0][1]) / 60))


def read_curve(run_dir):
    """-> [(step, reward)] from the run's tensorboard events ([] if none yet)."""
    from tensorboard.backend.event_processing import event_accumulator as ea

    acc = ea.EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    acc.Reload()
    if REWARD_TAG not in acc.Tags().get("scalars", []):
        return []
    return [(int(e.step), round(float(e.value), 2)) for e in acc.Scalars(REWARD_TAG)]


def clip_name(step: int) -> str:
    return f"clip_{step}.mp4"


def build_status(run_dir, clips_dir, total_steps=None, now=None, curve=None):
    ckpts = list_checkpoints(run_dir)
    curve = read_curve(run_dir) if curve is None else curve
    clips = []
    for i, (step, t, _) in enumerate(ckpts):
        if (Path(clips_dir) / clip_name(step)).is_file():
            clips.append({"step": step, "minutes": minutes_since_start(ckpts, i), "file": clip_name(step)})
    latest = ckpts[-1][0] if ckpts else 0
    return {
        "run": Path(run_dir).name,
        "started_at": ckpts[0][1] if ckpts else None,
        "latest_step": latest,
        "latest_minutes": minutes_since_start(ckpts, len(ckpts) - 1) if ckpts else 0,
        "practice_days": round(practice_days(latest), 1),
        "total_steps": total_steps,
        "curve": curve,
        "clips": clips,
        "updated_at": time.time() if now is None else now,
    }

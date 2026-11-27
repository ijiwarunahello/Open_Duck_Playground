"""Watch a training run: render a clip for each new checkpoint and keep status.json fresh.

Run with cwd = repository root:
    python tools/live_board/watcher.py --run checkpoints/<run> --out .tmp/live_board [--total-steps N]
Clips are rendered in a child process (tools/live_board/clip.py) so a rendering crash
never stops the watcher. status.json is rewritten every poll as a heartbeat.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from status import build_status, clip_name, list_checkpoints  # noqa: E402

SETTLE_S = 15  # an ONNX younger than this may still be being written
MAX_TRIES = 2
RENDER_ENV = {"MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl", "GALLIUM_DRIVER": "d3d12"}


def write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False))
    os.replace(tmp, path)


def pending(run_dir, clips_dir, failed, now):
    out = []
    for step, _, onnx in list_checkpoints(run_dir):
        if (clips_dir / clip_name(step)).is_file() or failed.get(step, 0) >= MAX_TRIES:
            continue
        if now - os.path.getmtime(onnx) < SETTLE_S:
            continue
        out.append(step)
    return out


def render_clip(run_dir, clips_dir, step) -> bool:
    cmd = [sys.executable, str(HERE / "clip.py"), "--run", str(run_dir),
           "--out", str(clips_dir / clip_name(step)), "--upto", str(step)]
    r = subprocess.run(cmd, env={**os.environ, **RENDER_ENV}, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[watcher] clip {step} failed:\n{r.stderr[-2000:]}", flush=True)
    return r.returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default=".tmp/live_board")
    ap.add_argument("--total-steps", type=int)
    ap.add_argument("--poll", type=float, default=20.0)
    ap.add_argument("--once", action="store_true", help="process what is there, then exit")
    a = ap.parse_args()
    run_dir, out = Path(a.run).resolve(), Path(a.out).resolve()
    clips = out / "clips"
    clips.mkdir(parents=True, exist_ok=True)
    failed = {}
    while True:
        todo = pending(run_dir, clips, failed, time.time())
        for step in todo:
            t0 = time.time()
            ok = render_clip(run_dir, clips, step)
            if not ok:
                failed[step] = failed.get(step, 0) + 1
            print(f"[watcher] step {step}: {'ok' if ok else 'FAILED'} ({time.time() - t0:.0f}s)", flush=True)
            write_json(out / "status.json", build_status(run_dir, clips, a.total_steps))
        try:
            write_json(out / "status.json", build_status(run_dir, clips, a.total_steps))
        except Exception as e:  # events file mid-write etc.: keep going, next poll retries
            print(f"[watcher] status failed: {e!r}", flush=True)
        if a.once:
            return
        time.sleep(a.poll)


if __name__ == "__main__":
    main()

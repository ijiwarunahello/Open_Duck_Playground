"""One short clip for the live board: up to 5 checkpoints of the same run race side by side.

The video carries no text. Next to it, clip_<step>.json holds each lane's training time and
its sign anchor per frame (fractions of the frame size), and board.html draws the signs, so
wording and script change in the page alone.

Lanes are the first checkpoint, the latest one and evenly spaced ones between, so each new
clip shows "how far it has come". Same start, command and push for every lane, as in the
race video (tools/race_video). Run with cwd = repository root and
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl GALLIUM_DRIVER=d3d12:
    python tools/live_board/clip.py --run checkpoints/<run> --out clip.mp4 [--upto STEP]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "race_video"))
from status import list_checkpoints, minutes_since_start  # noqa: E402

W, H, FPS, DT = 1280, 720, 30, 0.002
# Same timeline as the race video: stand 10 s, go, push 4.25 s later. Only T_SHOW.. is recorded.
T_SHOW, T_GO, T_PUSH, T_END = 8.5, 10.0, 14.25, 20.0
PUSH_VEL = (0.0, 0.8)
GOAL_X = 1.2
SPACING = 0.75
N_LANES = 5


def select_lanes(n_ckpts: int, n: int = N_LANES):
    """Indices: first, last and evenly spaced ones between (no duplicates)."""
    if n_ckpts <= n:
        return list(range(n_ckpts))
    return sorted({round(i * (n_ckpts - 1) / (n - 1)) for i in range(n)})


def simulate(lanes):
    from render_race import race_cmd

    n_frames = int(T_END * FPS) + 1
    first = int(T_SHOW * FPS)
    go, push = round(T_GO / DT), round(T_PUSH / DT)
    ys = [float(ln.data.qpos[1]) for ln in lanes]
    frames, step = [], 0
    for fi in range(n_frames):
        target = round(fi / FPS / DT)
        while step < target:
            if step == push:
                for ln in lanes:
                    ln.push(*PUSH_VEL)
            for ln, ly in zip(lanes, ys):
                ln.step(race_cmd(ln.data.qpos, GOAL_X, ly) if step >= go else (0.0, 0.0, 0.0))
            step += 1
        if fi >= first:
            frames.append([ln.qpos() for ln in lanes])
    return np.array(frames)


def camera(n):
    return dict(lookat=(GOAL_X / 2 + 0.05, (n - 1) * SPACING / 2, 0.05), distance=2.2 + 0.25 * n,
                azimuth=180.0, elevation=-24.0)


def sign_tracks(frames, n, cam, fovy, project):
    """Per frame, per lane: sign anchor (u/W, v/H) above the duck, or None off screen.
    Odd lanes sit higher so neighbouring signs never overlap."""
    out = []
    for q in frames:
        row = []
        for i in range(n):
            lift = 0.28 if i % 2 == 0 else 0.46
            u, v = project(q[i, :3] + [0, 0, lift], cam, W, H, fovy)
            row.append([round(u / W, 4), round(v / H, 4)] if 0 < u < W and 0 < v < H else None)
        out.append(row)
    return out


def meta_path(out_path) -> Path:
    return Path(out_path).with_suffix(".json")


def render(spec, out_path):
    """spec: [(onnx path, minutes since training start)] left to right on screen."""
    import imageio_ffmpeg
    import mediapy

    from race_sim import load_lanes
    from render_race import Stage, project

    lanes = load_lanes([(p, str(m)) for p, m in spec], spacing=SPACING)
    frames = simulate(lanes)
    n = len(lanes)
    stage = Stage(lanes[0].model, W, H)
    cam, lines = camera(n), course_lines(n)
    out = [stage.draw(q, cam, lines)[:, :, :3] for q in frames]
    meta = {"fps": FPS, "minutes": [m for _, m in spec], "signs": sign_tracks(frames, n, cam, stage.fovy, project)}
    stage.close()
    # json first: the watcher treats the mp4 as "done", so it must appear last
    tmp = meta_path(out_path).with_suffix(".tmp")
    tmp.write_text(json.dumps(meta, separators=(",", ":")))
    tmp.replace(meta_path(out_path))
    mediapy.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
    tmpv = Path(out_path).with_suffix(".tmp.mp4")
    mediapy.write_video(str(tmpv), out, fps=FPS)
    tmpv.replace(out_path)


def course_lines(n):
    y0, y1 = -SPACING / 2, (n - 0.5) * SPACING
    cy, hy = (y0 + y1) / 2, (y1 - y0) / 2
    lines = [((-0.12, cy, 0.001), (0.01, hy, 0.001), (1, 1, 1, 1)),
             ((GOAL_X, cy, 0.001), (0.015, hy, 0.001), (0.2, 0.75, 0.45, 1))]
    for i in range(n + 1):
        y = (i - 0.5) * SPACING
        lines.append(((GOAL_X / 2 + 0.1, y, 0.001), (GOAL_X / 2 + 0.4, 0.004, 0.0008), (0.85, 0.88, 0.9, 1)))
    return lines


def spec_for(run_dir, upto=None):
    ck = list_checkpoints(run_dir)
    if upto is not None:
        ck = [c for c in ck if c[0] <= upto]
    if not ck:
        raise SystemExit("no checkpoints")
    return [(ck[i][2], minutes_since_start(ck, i)) for i in select_lanes(len(ck))]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--upto", type=int, help="latest step to include (default: all)")
    a = ap.parse_args()
    s = spec_for(a.run, a.upto)
    print("lanes (minutes):", [m for _, m in s], flush=True)
    render(s, a.out)
    print("wrote", a.out)

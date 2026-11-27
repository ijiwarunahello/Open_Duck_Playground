"""One short clip for the live board: up to 5 checkpoints of the same run race side by side.

Lanes are the first checkpoint, the latest one and evenly spaced ones between, so each new
clip shows "how far it has come". Same start, command and push for every lane, as in the
race video (tools/race_video). Run with cwd = repository root and
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl GALLIUM_DRIVER=d3d12:
    python tools/live_board/clip.py --run checkpoints/<run> --out clip.mp4 [--upto STEP]
"""
import argparse
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
LABEL_PX = 26
SPACING = 0.75
N_LANES = 5


def select_lanes(n_ckpts: int, n: int = N_LANES):
    """Indices: first, last and evenly spaced ones between (no duplicates)."""
    if n_ckpts <= n:
        return list(range(n_ckpts))
    return sorted({round(i * (n_ckpts - 1) / (n - 1)) for i in range(n)})


def lane_label(minutes: int) -> str:
    from texts import clock_text

    return clock_text(minutes, "kanji").split(" ", 1)[1]


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


def render(spec, out_path):
    """spec: [(onnx path, label)] left to right on screen."""
    import imageio_ffmpeg
    import mediapy
    from PIL import Image, ImageDraw

    from race_sim import load_lanes
    from render_race import Overlay, Stage, project

    lanes = load_lanes(spec, spacing=SPACING)
    frames = simulate(lanes)
    n = len(lanes)
    stage = Stage(lanes[0].model, W, H)
    cam, lines, ov = camera(n), course_lines(n), Overlay()
    out = []
    for q in frames:
        img = Image.fromarray(stage.draw(q, cam, lines))
        dr = ImageDraw.Draw(img, "RGBA")
        for i, (_, label) in enumerate(spec):
            lift = 0.28 if i % 2 == 0 else 0.46  # stagger so neighbouring signs never overlap
            u, v = project(q[i, :3] + [0, 0, lift], cam, W, H, stage.fovy)
            if 0 < u < W and 0 < v < H:
                ov.panel(dr, label, (u, v), LABEL_PX)
        out.append(np.asarray(img)[:, :, :3])
    stage.close()
    mediapy.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
    tmp = Path(out_path).with_suffix(".tmp.mp4")
    mediapy.write_video(str(tmp), out, fps=FPS)
    tmp.replace(out_path)


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
    return [(ck[i][2], lane_label(minutes_since_start(ck, i))) for i in select_lanes(len(ck))]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--upto", type=int, help="latest step to include (default: all)")
    a = ap.parse_args()
    s = spec_for(a.run, a.upto)
    print("lanes:", [lb for _, lb in s], flush=True)
    render(s, a.out)
    print("wrote", a.out)

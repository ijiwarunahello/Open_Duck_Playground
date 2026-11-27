"""Render "みんなで れんしゅう": the same crowd of ducks at five training stages.

Every stage uses the same ducks, seeds, commands and pushes (training conditions, see
crowd.py) and only swaps the checkpoint. Fall counts on screen and in the bar chart are
the counts from these runs.

Run with cwd = repository root and
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl GALLIUM_DRIVER=d3d12:
    python tools/race_video/crowd_video.py --out crowd.mp4
"""
import argparse
import multiprocessing as mp
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lanes import CKPT  # noqa: E402

W, H, FPS = 1920, 1080, 30
T_TITLE, STAGE_S, T_END = 4.0, 8.0, 50.0
COUNT_UP_S = 0.8
FADE_S = 0.5
SIDE = 20  # 400 ducks
WARMUP_S = 3.0
SEED = 7
CAM = dict(lookat=(0.0, 0.6, 0.0), distance=5.5, azimuth=125.0, elevation=-30.0)

TITLE = "みんなで れんしゅう"
SUMMARY_TITLE = "ころんだ かず"
SUMMARY_LINE = "8192たいで 3じかん = 69にちぶんの れんしゅう"

# (checkpoint, minutes shown). 3 hours = run end (3 h 6 min wall clock).
STAGES = [
    (CKPT / "2026_11_27_014427_0.onnx", 0),
    (CKPT / "2026_11_27_021146_42926080.onnx", 27),
    (CKPT / "2026_11_27_022458_64389120.onnx", 40),
    (CKPT / "2026_11_27_031742_150241280.onnx", 97),
    (CKPT / "2026_11_27_045005_300482560.onnx", 180),
]


def clock_text(minutes: int) -> str:
    h, m = divmod(int(minutes), 60)
    if m == 0 and h:
        return f"れんしゅう {h}じかん"
    unit = "ふん" if m % 10 in (2, 5, 7, 9) else "ぷん"
    return f"れんしゅう {h}じかん{m}{unit}" if h else f"れんしゅう {m}{unit}"


def stage_at(t: float) -> int:
    return int(min(max((t - T_TITLE) // STAGE_S, 0), len(STAGES) - 1))


def clock_minutes(t: float) -> int:
    i = stage_at(t)
    if t < T_TITLE or i == 0:
        return 0
    into = t - (T_TITLE + i * STAGE_S)
    prev, cur = STAGES[i - 1][1], STAGES[i][1]
    if into >= COUNT_UP_S:
        return cur
    return int(prev + (cur - prev) * max(into, 0.0) / COUNT_UP_S)


def bar_heights(counts, max_px):
    top = max(counts) or 1
    return [int(round(c / top * max_px)) for c in counts]


# ---------- rendering (frames split across processes, then concatenated) ----------

_DATA = {}


def _frame(k, stage, ov):
    from PIL import Image, ImageDraw

    t = k / FPS
    if t >= T_TITLE + len(STAGES) * STAGE_S:
        return _summary(ov, t)
    i = stage_at(t)
    frames, falls = _DATA["frames"][i], _DATA["falls"][i]
    if t < T_TITLE:
        q, n_fall = frames[0], None
    else:
        fi = min(int(round((t - T_TITLE - i * STAGE_S) * FPS)), len(frames) - 1)
        q, n_fall = frames[fi], int(falls[fi])
    img = Image.fromarray(stage.draw(q, CAM))
    dr = ImageDraw.Draw(img, "RGBA")
    if t < T_TITLE:
        ov.panel(dr, TITLE, (W / 2, H / 2), 120)
    else:
        text = clock_text(clock_minutes(t))
        tw = dr.textlength(text, font=ov.font(64))
        ov.panel(dr, text, (60 + tw / 2 + 21, 80), 64)
        label = f"ころんだ {n_fall}"
        lw = dr.textlength(label, font=ov.font(64))
        ov.panel(dr, label, (W - 60 - lw / 2 - 21, 80), 64, fg=(255, 255, 255), bg=(226, 85, 27, 240))
    return np.asarray(img)[:, :, :3]


def _summary(ov, t):
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (W, H), (40, 48, 45))
    dr = ImageDraw.Draw(img, "RGBA")
    ov.panel(dr, SUMMARY_TITLE, (W / 2, 110), 80)
    counts = [int(f[-1]) for f in _DATA["falls"]]
    heights = bar_heights(counts, 520)
    n = len(counts)
    bw, gap = 200, 110
    x0 = (W - (n * bw + (n - 1) * gap)) / 2
    base = 860
    for j, (c, hpx) in enumerate(zip(counts, heights)):
        x = x0 + j * (bw + gap)
        dr.rectangle((x, base - hpx, x + bw, base), fill=(226, 85, 27, 255))
        dr.text((x + bw / 2, base - hpx - 16), str(c), font=ov.font(56), fill=(255, 255, 255), anchor="mb")
        dr.text((x + bw / 2, base + 22), clock_text(STAGES[j][1]).replace("れんしゅう ", ""), font=ov.font(44),
                fill=(235, 240, 238), anchor="mt")
    if t >= T_TITLE + len(STAGES) * STAGE_S + 1.5:
        ov.panel(dr, SUMMARY_LINE, (W / 2, 1000), 64)
    return np.asarray(img)


def _render_range(args):
    k0, k1, out_path = args
    import mediapy
    import imageio_ffmpeg

    sys.stdout = open(os.devnull, "w")
    from race_sim import Lane
    from render_race import Overlay, Stage

    model = Lane(str(STAGES[0][0]), "", 0.0).model
    stage = Stage(model, W, H, max_geom=SIDE * SIDE * 60)
    ov = Overlay()
    total = int(T_END * FPS)
    out = []
    for k in range(k0, k1):
        f = _frame(k, stage, ov)
        fade = min(1.0, (k + 1) / (FADE_S * FPS), (total - k) / (FADE_S * FPS))
        if fade < 1.0:
            f = (f.astype(np.float32) * fade).astype(np.uint8)
        out.append(f)
    stage.close()
    mediapy.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
    mediapy.write_video(out_path, out, fps=FPS)
    return out_path


def render(out_path, workers=6):
    import imageio_ffmpeg

    from crowd import simulate_crowd

    frames, falls = [], []
    for onnx, minutes in STAGES:
        print(f"simulating stage {minutes} min...", flush=True)
        fr, fa = simulate_crowd(str(onnx), SIDE, STAGE_S, warmup=WARMUP_S, seed=SEED)
        frames.append(fr)
        falls.append(fa)
        print(f"  falls in {STAGE_S:.0f} s: {fa[-1]}", flush=True)
    _DATA["frames"], _DATA["falls"] = frames, falls

    total = int(T_END * FPS)
    bounds = np.linspace(0, total, workers + 1).astype(int)
    tmp = tempfile.mkdtemp(prefix="crowd_video_")
    jobs = [(int(a), int(b), os.path.join(tmp, f"part{j:02d}.mp4")) for j, (a, b) in enumerate(zip(bounds[:-1], bounds[1:]))]
    with mp.get_context("fork").Pool(workers) as pool:
        parts = pool.map(_render_range, jobs)
    listing = os.path.join(tmp, "parts.txt")
    with open(listing, "w") as f:
        f.writelines(f"file '{p}'\n" for p in parts)
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                    "-i", listing, "-c", "copy", out_path], check=True)
    print(f"wrote {out_path}; falls per stage: {[int(f[-1]) for f in falls]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="crowd.mp4")
    render(ap.parse_args().out)

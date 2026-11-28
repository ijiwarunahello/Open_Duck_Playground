"""Render "アヒルは どうやって 歩けるように なったの?" (storyboard: 6 panels, 60 s loop, kanji + ruby).

Run with cwd = repository root and
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl GALLIUM_DRIVER=d3d12:
    python tools/learning_video/render_learning.py --out learning.mp4
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "race_video"))
from curve import load_curve, partial_polyline, to_pixels  # noqa: E402
from learn_texts import T  # noqa: E402
from scored import CARD_FULL, CARDS, card_values  # noqa: E402

W, H, FPS, DT = 1920, 1080, 30, 0.002

# video timeline (seconds): scene k runs SCENES[k-1] .. SCENES[k]
SCENES = (0.0, 4.0, 16.0, 30.0, 44.0, 53.0, 60.0)
FADE_S = 0.5

WARMUP_S = 2.0  # sim seconds walked before the first frame of scenes 1-3
# Scene 5 push, picked with push_pick.py (2026-11-29): 0.8 m/s sideways (training range 0.1-1.0)
# 3.75 s after the start; the 3-hour duck tilts ~25 deg and recovers. At 1.0 m/s it fell in 3 of 5 tries.
PUSH_T = 3.75
PUSH_VEL = (0.0, 0.8)
S5_NORMAL_BEFORE = 2.0  # video seconds at normal speed before the slow part
S5_SLOW = (3.5, 5.5)  # sim window shown at SLOW_SPEED
SLOW_SPEED = 0.4
GHOST_DY = 0.32  # the model duck walks this far to the duck's left (further from the scene-3 camera)
GHOST_RGBA = (0.42, 0.64, 0.79, 0.35)
EMA_TAU = 0.25  # s, smoothing of the scene-3 bars
THUMB = 240  # px, scene-4 windows
THUMB_CX = (420, 760, 1620)  # window centres (fixed row above the chart)
THUMB_TOP = 200

INK = (29, 36, 39)
TELOP = (226, 85, 27)
GOOD = (47, 143, 91)
BAD = (194, 65, 45)
GHOST = (106, 163, 201)
DUCK = (242, 194, 48)
PAPER = (238, 242, 244)
RULE = (201, 211, 215)
TELOP_PX = 64
LABEL_PX = 44


# ---------- pure helpers (tested) ----------

def scene_at(t: float) -> int:
    for k in range(1, len(SCENES)):
        if t < SCENES[k]:
            return k
    return len(SCENES) - 1


def s5_sim_time(v: float) -> float:
    """Video seconds since scene 5 start -> sim seconds of the push run.
    Normal speed up to S5_SLOW[0], slow through the window, then normal again."""
    start = S5_SLOW[0] - S5_NORMAL_BEFORE
    if v < S5_NORMAL_BEFORE:
        return start + v
    slow_len = (S5_SLOW[1] - S5_SLOW[0]) / SLOW_SPEED
    if v < S5_NORMAL_BEFORE + slow_len:
        return S5_SLOW[0] + (v - S5_NORMAL_BEFORE) * SLOW_SPEED
    return S5_SLOW[1] + (v - S5_NORMAL_BEFORE - slow_len)


def bar_frac(v: float, full: float) -> float:
    return float(min(max(v / full, 0.0), 1.0))


def ema(series, tau=EMA_TAU, dt=1 / FPS):
    a = 1 - math.exp(-dt / tau)
    out, acc = [], None
    for x in series:
        acc = x if acc is None else acc + a * (x - acc)
        out.append(acc)
    return out


def yaw_quat(qpos):
    """Upright quaternion (wxyz) keeping only the yaw of qpos[3:7]."""
    w, x, y, z = qpos[3:7]
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


def ghost_qpos(duck_qpos, ref, joint_addr, stand_z, dy=GHOST_DY):
    """Model duck: the duck's x, offset sideways, standing height, upright with the duck's yaw,
    and the reference joint angles (ref[0:16] = left leg 5, neck/head 4, antennas 2, right leg 5;
    antennas dropped). The reference frame carries no body pose we can use (its "quat" slice 3:7
    overlaps the joint angles; joystick.py leaves that reward term out), so the body stays upright."""
    q = np.array(duck_qpos, float).copy()
    q[1] += dy
    q[2] = stand_z
    q[3:7] = yaw_quat(duck_qpos)
    joints = np.concatenate([ref[0:5], ref[5:9], ref[11:16]])
    q[np.asarray(joint_addr)] = joints
    return q


# ---------- simulation ----------

def sim_main(onnx, seconds):
    """Scenes 1-3: one 3-hour duck walking forward. Per video frame: qpos, obs, action, card values, ghost qpos."""
    from render_race import heading_cmd
    from scored import ScoredLane

    ln = ScoredLane(onnx, "", 0.0)
    s = ln.inf
    addr = s.get_actuator_joints_addr()
    stand_z = float(ln.data.qpos[2])
    rec = {"qpos": [], "obs": [], "action": [], "cards": [], "ghost": [], "gravity": []}
    step = 0
    n_warm = round(WARMUP_S / DT)
    for fi in range(int(seconds * FPS) + 1):
        target = n_warm + round(fi / FPS / DT)
        while step < target:
            ln.step(heading_cmd(ln.data.qpos, lane_y=0.0))
            step += 1
        last = ln.last
        rec["qpos"].append(ln.qpos())
        rec["obs"].append(last["obs"])
        rec["action"].append(last["action"])
        rec["gravity"].append(last["gravity"])
        rec["cards"].append(card_values(last["raw"]))
        rec["ghost"].append(ghost_qpos(ln.qpos(), last["ref"], addr, stand_z))
    assert not ln.fallen, "3-hour duck fell in the main walk"
    return rec


def sim_push(onnx, seconds):
    """Scenes 5-6: walk, push sideways at PUSH_T. qpos at every physics step."""
    from race_sim import Lane
    from render_race import heading_cmd

    ln = Lane(onnx, "", 0.0)
    qs = [ln.qpos()]
    for k in range(int(seconds / DT)):
        if k == round(PUSH_T / DT):
            ln.push(*PUSH_VEL)
        ln.step(heading_cmd(ln.data.qpos, lane_y=0.0))
        qs.append(ln.qpos())
    assert not ln.fallen, "3-hour duck fell after the push; re-run push_pick.py"
    return np.array(qs)


def sim_thumbs(onnxes, seconds):
    from race_sim import Lane
    from render_race import heading_cmd

    out = []
    for onnx in onnxes:
        ln = Lane(onnx, "", 0.0)
        frames, step = [], 0
        for fi in range(int(seconds * FPS) + 1):
            while step < round(fi / FPS / DT):
                ln.step(heading_cmd(ln.data.qpos, lane_y=0.0))
                step += 1
            frames.append(ln.qpos())
        out.append(np.array(frames))
    return out


# ---------- drawing ----------

def make_ghost_stage(model, w, h):
    from render_race import Stage

    class GhostStage(Stage):
        def draw_with_ghost(self, q, ghost, cam):
            mj = self.mj
            c = mj.MjvCamera()
            c.type = mj.mjtCamera.mjCAMERA_FREE
            c.lookat[:] = cam["lookat"]
            c.distance, c.azimuth, c.elevation = cam["distance"], cam["azimuth"], cam["elevation"]
            self.d.qpos[:] = q
            mj.mj_forward(self.m, self.d)
            self.r.update_scene(self.d, camera=c)
            scn = self.r.scene
            self.d.qpos[:] = ghost
            mj.mj_forward(self.m, self.d)
            n0 = scn.ngeom
            mj.mjv_addGeoms(self.m, self.d, self.opt, self.pert, mj.mjtCatBit.mjCAT_DYNAMIC, scn)
            for i in range(n0, scn.ngeom):
                scn.geoms[i].rgba[:] = GHOST_RGBA
            return self.r.render()

    return GhostStage(model, w, h)


def cam_follow(q, distance, azimuth, elevation, dz=0.10, dx=0.0):
    return dict(lookat=(float(q[0]) + dx, float(q[1]), dz), distance=distance, azimuth=azimuth, elevation=elevation)


def top_telop(ov, dr, text, size=TELOP_PX, y=36, fg=INK):
    bh = ov.box_height(text, size)
    ov.panel(dr, text, (W / 2, y + bh / 2), size, fg=fg)
    return y + bh


def hbar(dr, x, y, w, h, frac, color, centre=False):
    dr.rounded_rectangle((x, y, x + w, y + h), radius=h // 3, fill=(255, 255, 255, 230))
    if centre:
        mid = x + w / 2
        end = mid + frac * w / 2
        lo, hi = sorted((mid, end))
        dr.rectangle((lo, y + 3, max(hi, lo + 2), y + h - 3), fill=color)
        dr.line((mid, y - 4, mid, y + h + 4), fill=INK, width=2)
    else:
        dr.rounded_rectangle((x + 3, y + 3, x + 3 + max(frac * (w - 6), 2), y + h - 3), radius=h // 3, fill=color)


def label(ov, dr, text, left, cy, size=LABEL_PX, fg=INK, bg=(255, 255, 255, 0)):
    tw = ov.measure(text, size)[0]
    ov.panel(dr, text, (left + tw / 2 + size // 3, cy), size, fg=fg, bg=bg)


def draw_scene2(ov, dr, rec_i, t_local):
    obs, act, grav = rec_i["obs"], rec_i["action"], rec_i["gravity"]
    # inputs (left column)
    x0, w = 70, 380
    label(ov, dr, T["tilt"], x0 - 14, 250)
    hbar(dr, x0, 290, w, 34, float(np.clip(grav[1] / 0.08, -1, 1)), GHOST, centre=True)
    label(ov, dr, T["joints"], x0 - 14, 400)
    legs = np.concatenate([obs[9:14], obs[18:23]])  # joint angle - default, legs only
    bw = w / len(legs)
    for i, v in enumerate(legs):
        hgt = float(np.clip(v / 0.3, -1, 1)) * 60
        cx = x0 + i * bw
        dr.rectangle((cx + 4, 490 - max(hgt, 0), cx + bw - 4, 490 - min(hgt, 0)), fill=GHOST)
    dr.line((x0, 490, x0 + w, 490), fill=INK, width=2)
    label(ov, dr, T["cmd"], x0 - 14, 590)
    cmd = obs[6:9]
    ax, ay = x0 + 60, 660
    dr.line((ax, ay, ax + 220 * cmd[0] / 0.15, ay), fill=GHOST, width=14)
    dr.polygon([(ax + 220 * cmd[0] / 0.15 + 30, ay), (ax + 220 * cmd[0] / 0.15, ay - 22), (ax + 220 * cmd[0] / 0.15, ay + 22)], fill=GHOST)
    # brain
    bx0, by0, bx1, by1 = 760, 200, 1160, 360
    pulse = 0.5 + 0.5 * math.sin(t_local * 2 * math.pi * 2.0)
    dr.rounded_rectangle((bx0, by0, bx1, by1), radius=28, fill=(255, 255, 255, 240), outline=INK, width=5)
    tw = ov.measure(T["brain"], 56)[0]
    ov.panel(dr, T["brain"], ((bx0 + bx1) / 2, by0 + 62), 56, bg=(0, 0, 0, 0))
    ov.panel(dr, "101 → 14", ((bx0 + bx1) / 2, by1 - 44), 40, fg=(91, 106, 112), bg=(0, 0, 0, 0))
    # flowing dots along connectors (left -> brain -> right)
    for k in range(4):
        a = ((t_local * 1.5) + k / 4) % 1.0
        px = 470 + a * (bx0 - 470)
        dr.ellipse((px - 9, 450 - 9 - a * 170, px + 9, 450 + 9 - a * 170), fill=TELOP)
        qx = bx1 + a * (1450 - bx1)
        dr.ellipse((qx - 9, 280 - 9 + a * 170, qx + 9, 280 + 9 + a * 170), fill=TELOP)
    dr.rounded_rectangle((bx0 - 8, by0 - 8, bx1 + 8, by1 + 8), radius=34, outline=TELOP + (int(80 + 120 * pulse),), width=6)
    # motors (right column)
    mx0, mw = 1460, 400
    label(ov, dr, T["motors"], mx0 - 14, 250)
    n = len(act)
    bw = mw / n
    base = 520
    for i, v in enumerate(act):
        hgt = float(np.clip(v, -1, 1)) * 150
        cx = mx0 + i * bw
        dr.rectangle((cx + 3, base - max(hgt, 0), cx + bw - 3, base - min(hgt, 0)), fill=DUCK, outline=INK, width=2)
    dr.line((mx0, base, mx0 + mw, base), fill=INK, width=2)


def draw_cards(ov, dr, vals, lit):
    names = [n for n, _ in CARDS]
    cw, ch, gap = 400, 230, 40
    x = (W - (4 * cw + 3 * gap)) / 2
    y = H - ch - 50
    for i, name in enumerate(names):
        on = i < lit
        alpha = 245 if on else 110
        dr.rounded_rectangle((x, y, x + cw, y + ch), radius=22, fill=(255, 255, 255, alpha))
        txt_fg = INK if on else (140, 150, 154)
        ov.panel(dr, T["cards"][name], (x + cw / 2, y + 56), 44, fg=txt_fg, bg=(0, 0, 0, 0))
        sign, col = ("－", BAD) if name == "jitter" else ("＋", GOOD)
        f = ov.font(70)
        dr.text((x + cw / 2, y + 128), sign, font=f, fill=col if on else (200, 205, 207), anchor="mm")
        if on:
            full = CARD_FULL.get(name, JITTER_FULL)
            hbar(dr, x + 30, y + ch - 56, cw - 60, 32, bar_frac(vals[name], full), col)
        x += cw + gap


JITTER_FULL = 1.0  # set in render() from the run (99th percentile), so the bar uses its range


def draw_curve(ov, dr, px_all, frac, box, thumbs, thumb_imgs, x_max_min):
    l, t, r, b = box
    for yv in (0, 100, 200, 300):
        yy = b - yv / 350 * (b - t)
        dr.line((l, yy, r, yy), fill=RULE, width=2)
        f = ov.font(30)
        dr.text((l - 16, yy), str(yv), font=f, fill=(91, 106, 112), anchor="rm")
    for i, txt in enumerate(T["hours"]):
        xx = l + (i * 60) / x_max_min * (r - l)
        ov.panel(dr, txt, (xx, b + 44), 34, fg=(91, 106, 112), bg=(0, 0, 0, 0))
    ov.panel(dr, T["x_axis"], ((l + r) / 2, b + 110), 38, fg=(91, 106, 112), bg=(0, 0, 0, 0))
    ov.panel(dr, T["y_axis"], (l - 60, t - 40), 38, fg=(91, 106, 112), bg=(0, 0, 0, 0))
    pts = partial_polyline(px_all, frac)
    if len(pts) >= 2:
        dr.line(pts, fill=TELOP, width=10, joint="curve")
    ex, ey = pts[-1]
    dr.ellipse((ex - 12, ey - 12, ex + 12, ey + 12), fill=TELOP)
    return pts[-1]


# ---------- main ----------

def render(out_path, preview=None):
    import mediapy
    import imageio_ffmpeg
    from PIL import Image, ImageDraw

    from lanes import CKPT
    from render_race import Overlay, Stage, ease

    global JITTER_FULL
    best = str(CKPT / "2026_11_27_045005_300482560.onnx")
    thumb_ckpts = [str(CKPT / "2026_11_27_014427_0.onnx"), str(CKPT / "2026_11_27_022458_64389120.onnx"), best]

    print("simulating main walk...", flush=True)
    rec = sim_main(best, SCENES[3])
    rec_cards = {n: ema([c[n] for c in rec["cards"]]) for n, _ in CARDS}
    JITTER_FULL = float(np.percentile([c["jitter"] for c in rec["cards"]], 99))
    s3 = [i for i in range(len(rec["qpos"])) if SCENES[2] <= i / FPS < SCENES[3]]
    print("scene-3 mean card values (per step, unsmoothed):",
          {n: round(float(np.mean([rec["cards"][i][n] for i in s3])), 3) for n, _ in CARDS},
          "jitter full:", round(JITTER_FULL, 3), flush=True)
    print("simulating push...", flush=True)
    s6_len = SCENES[6] - SCENES[5]
    push_q = sim_push(best, s5_sim_time(SCENES[5] - SCENES[4]) + s6_len + 0.5)
    print("simulating thumbnails...", flush=True)
    thumbs = sim_thumbs(thumb_ckpts, SCENES[4] - SCENES[3])
    curve = load_curve(CKPT)
    print("curve:", [(round(m), round(v, 1)) for m, v in curve], flush=True)
    x_max = 190.0
    box = (260, 600, 1780, 900)
    px_all = to_pixels(curve, box, x_max, 350.0)
    # thumbnails sit at the checkpoints 0, 40 min and the last
    thumb_mins = [curve[0][0], min(curve, key=lambda p: abs(p[0] - 40))[0], curve[-1][0]]
    thumb_px = [px_all[[m for m, _ in curve].index(tm)] for tm in thumb_mins]

    lane_model = __import__("race_sim").Lane(best, "", 0.0).model
    stage = make_ghost_stage(lane_model, W, H)
    tstage = Stage(lane_model, THUMB, THUMB)
    ov = Overlay()

    mediapy.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
    total = int(SCENES[-1] * FPS)
    import contextlib
    ks = range(total) if preview is None else [int(round(x * FPS)) for x in preview]
    writer = contextlib.nullcontext() if preview is not None else mediapy.VideoWriter(out_path, shape=(H, W), fps=FPS)
    with writer as vw:
        for k in ks:
            t = k / FPS
            sc = scene_at(t)
            if sc <= 3:
                i = min(k, len(rec["qpos"]) - 1)
                q = rec["qpos"][i]
                if sc == 1:
                    a = t / SCENES[1]
                    cam = cam_follow(q, 1.15 - 0.12 * a, 165.0, -8.0, dz=0.20)
                    img = Image.fromarray(stage.draw([q], cam))
                    dr = ImageDraw.Draw(img, "RGBA")
                    ov.panel(dr, T["title"], (W / 2, 120), 84)
                elif sc == 2:
                    cam = cam_follow(q, 1.4, 90.0, -6.0, dz=0.32)
                    img = Image.fromarray(stage.draw([q], cam))
                    dr = ImageDraw.Draw(img, "RGBA")
                    dr.rectangle((0, 0, W, H), fill=(255, 255, 255, 60))
                    top_telop(ov, dr, T["loop"])
                    tl = t - SCENES[1]
                    draw_scene2(ov, dr, {"obs": rec["obs"][i], "action": rec["action"][i], "gravity": rec["gravity"][i]}, tl)
                    if tl > 3.0:
                        ov.panel(dr, T["rate"], (W / 2, H - 100), 80, fg=TELOP)
                else:
                    cam = dict(cam_follow(q, 1.25, 150.0, -12.0, dz=0.20), lookat=(float(q[0]), float(q[1]) + GHOST_DY / 2, 0.20))
                    img = Image.fromarray(stage.draw_with_ghost(q, rec["ghost"][i], cam))
                    dr = ImageDraw.Draw(img, "RGBA")
                    top_telop(ov, dr, T["score"], fg=TELOP)
                    u, v = __import__("render_race").project(q[:3] + [0, 0, 0.27], cam, W, H, stage.fovy)
                    ov.panel(dr, T["duck3h"], (u, v), 40)
                    g = rec["ghost"][i]
                    u, v = __import__("render_race").project(g[:3] + [0, 0, 0.27], cam, W, H, stage.fovy)
                    ov.panel(dr, T["model"], (u, v), 40, fg=(255, 255, 255), bg=GHOST + (235,))
                    lit = 1 + int((t - SCENES[2]) / 2.0)
                    draw_cards(ov, dr, {n: rec_cards[n][i] for n, _ in CARDS}, min(lit, 4))
                frame = np.asarray(img)
            elif sc == 4:
                tl = t - SCENES[3]
                frac = ease(min(tl / 9.0, 1.0))
                img = Image.new("RGB", (W, H), PAPER)
                dr = ImageDraw.Draw(img, "RGBA")
                top_telop(ov, dr, T["curve"], fg=TELOP, y=30)
                end = draw_curve(ov, dr, px_all, frac, box, thumbs, None, x_max)
                for j, (tp, tq) in enumerate(zip(thumb_px, thumbs)):
                    if end[0] + 1 < tp[0]:
                        continue
                    qj = tq[min(int(tl * FPS), len(tq) - 1)]
                    timg = Image.fromarray(tstage.draw([qj], cam_follow(qj, 0.85, 100.0, -6.0, dz=0.21)))
                    cx = THUMB_CX[j]
                    x0, y0 = int(cx - THUMB / 2), THUMB_TOP
                    dr.line((cx, y0 + THUMB + 80, tp[0], tp[1]), fill=INK, width=3)
                    dr.rounded_rectangle((x0 - 8, y0 - 8, x0 + THUMB + 8, y0 + THUMB + 8), radius=16, fill=INK)
                    img.paste(timg, (x0, y0))
                    dr = ImageDraw.Draw(img, "RGBA")
                    ov.panel(dr, T["thumbs"][j], (cx, y0 + THUMB + 44), 40)
                frame = np.asarray(img)
            elif sc == 5:
                v = t - SCENES[4]
                st = s5_sim_time(v)
                q = push_q[min(int(round(st / DT)), len(push_q) - 1)]
                cam = cam_follow(q, 1.2, 150.0, -16.0, dz=0.16)
                img = Image.fromarray(stage.draw([q], cam))
                dr = ImageDraw.Draw(img, "RGBA")
                top_telop(ov, dr, T["tease"])
                if PUSH_T - 0.15 <= st < PUSH_T + 0.6:
                    from render_race import project
                    p = q[:3] + [0, 0, 0.10]
                    a0 = project(p - [0, 0.36, 0], cam, W, H, stage.fovy)
                    a1 = project(p - [0, 0.12, 0], cam, W, H, stage.fovy)
                    ov.arrow(dr, a0, a1, width=16)
                    ov.shout(dr, T["push"], (a0[0], a0[1] - 110), 110)
                if S5_SLOW[0] <= st < S5_SLOW[1]:
                    ov.panel(dr, T["slow"], (W - 150, H - 80), 48, fg=(255, 255, 255), bg=(0, 0, 0, 150))
                frame = np.asarray(img)
            else:
                v = t - SCENES[5]
                st = s5_sim_time(SCENES[5] - SCENES[4]) + v
                q = push_q[min(int(round(st / DT)), len(push_q) - 1)]
                a = ease(v / (SCENES[6] - SCENES[5]))
                cam = cam_follow(q, 1.2 + 1.5 * a, 150.0 - 20.0 * a, -16.0 - 14.0 * a, dz=0.16)
                img = Image.fromarray(stage.draw([q], cam))
                dr = ImageDraw.Draw(img, "RGBA")
                if v >= 1.0:
                    top = H - 40 - sum(ov.box_height(x, TELOP_PX) + 14 for x in T["end"])
                    dr.rectangle((0, top - 20, W, H), fill=(0, 0, 0, 90))
                    ov.stack_up(dr, T["end"], TELOP_PX, H - 40)
                frame = np.asarray(img)
            fade = min(1.0, (k + 1) / (FADE_S * FPS), (total - k) / (FADE_S * FPS))
            if fade < 1.0:
                frame = (frame.astype(np.float32) * fade).astype(np.uint8)
            if preview is not None:
                Image.fromarray(frame[:, :, :3]).save(f"{out_path}_{k / FPS:05.1f}.png")
            else:
                vw.add_image(np.ascontiguousarray(frame[:, :, :3]))
            if k % 150 == 0:
                print(f"frame {k}/{total}", flush=True)
    stage.close()
    tstage.close()
    print(f"wrote {out_path} ({total} frames)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="learning.mp4")
    ap.add_argument("--preview", help="comma-separated video times; writes <out>_<t>.png instead of the mp4")
    args = ap.parse_args()
    render(args.out, [float(x) for x in args.preview.split(",")] if args.preview else None)

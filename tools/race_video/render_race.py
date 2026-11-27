"""Render the "アヒルの かけっこ" race video (storyboard: 6 panels, ~45 s loop).

Run with cwd = repository root and
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl GALLIUM_DRIVER=d3d12:
    python tools/race_video/render_race.py --out race.mp4
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

W, H, FPS, DT = 1920, 1080, 30, 0.002
FONT = "/mnt/c/Windows/Fonts/BIZ-UDGothicB.ttc"
SPACING = 0.55  # with lane-centre steering the closest approach is ~0.45 m below spacing

# video timeline (seconds)
T_PAN, T_GO, T_RACE, T_RACE_END, T_REPLAY_END, T_RESULTS_END, T_END = 4.0, 10.0, 12.0, 30.0, 38.0, 42.0, 52.0
CROWD_TEXT = "8192たいで いっしょに れんしゅう"
CROWD_SIDE = 20  # 20 x 20 = 400 ducks
CROWD_WARMUP_S = 5.0
CROWD_SEED = 42
from lanes import CKPT as _CKPT  # noqa: E402
CROWD_CKPT = _CKPT / "2026_11_27_023811_85852160.onnx"  # 54 minutes: walks, still falls sometimes
REPLAY_SIM = (13.6, 16.8)  # sim window shown in slow motion (push happens at 14.25)
REPLAY_SPEED = 0.4
FADE_S = 0.5

TITLE = "アヒルの かけっこ"
GO_TEXT = "よーい、ドン!"
PUSH_TEXT = "えいっ!"
END_LINES = ("3じかんで 69にちぶん れんしゅうしたよ", "となりの duck も、", "こうして れんしゅうした AI で あるいています")
LABEL_PX = 40  # lane signs (plan floor: 40 px)
END_PX = 64  # closing lines (plan floor: 64 px)


# ---------- pure helpers (tested) ----------

def _cam_basis(cam):
    az, el = math.radians(cam["azimuth"]), math.radians(cam["elevation"])
    f = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    r = np.cross(f, [0.0, 0.0, 1.0])
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    pos = np.asarray(cam["lookat"], float) - cam["distance"] * f
    return pos, f, r, u


def project(p, cam, width, height, fovy):
    """World point -> pixel (u, v) for a MuJoCo free camera."""
    pos, f, r, u = _cam_basis(cam)
    v = np.asarray(p, float) - pos
    z = v @ f
    focal = (height / 2) / math.tan(math.radians(fovy) / 2)
    return width / 2 + focal * (v @ r) / z, height / 2 - focal * (v @ u) / z


def result_label(fallen, dx, goal_x):
    if fallen:
        return "ころんだ"
    if dx >= goal_x:
        return "ゴール!"
    if dx < 0.2:
        return "うごかない"
    return "とちゅう"


HEADING_GAIN = 2.0  # rad/s of turn command per rad of heading error
LANE_GAIN = 1.0  # rad of target heading per metre off the lane centre
MAX_LANE_HEADING = 0.4  # rad


def heading_cmd(qpos, speed=0.15, lane_y=None):
    """Forward command plus a turn command that steers back toward +x and, if lane_y is
    given, toward the lane centre (what an operator does with the stick). Same rule for
    every lane."""
    w, x, y, z = qpos[3:7]
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    target = 0.0
    if lane_y is not None:
        target = float(np.clip(-LANE_GAIN * (qpos[1] - lane_y), -MAX_LANE_HEADING, MAX_LANE_HEADING))
    return (speed, 0.0, float(np.clip(HEADING_GAIN * (target - yaw), -1.0, 1.0)))


STOP_PAST_GOAL = 0.3  # metres past the goal line where a duck is told to stop


def race_cmd(qpos, goal_x, lane_y):
    if qpos[0] >= goal_x + STOP_PAST_GOAL:
        return (0.0, 0.0, 0.0)
    return heading_cmd(qpos, lane_y=lane_y)


# ---------- simulation ----------

def simulate(lanes, push_time, push_vel, goal_x=1.2):
    """Returns per-video-frame qpos (n_frames, n_lanes, nq), fallen flags, and
    per-physics-step qpos inside REPLAY_SIM for smooth slow motion."""
    n_frames = int(T_RACE_END * FPS) + 1
    go_step = round(T_GO / DT)
    push_step = round((T_GO + push_time) / DT)
    r0, r1 = round(REPLAY_SIM[0] / DT), round(REPLAY_SIM[1] / DT)
    frames, fallen, replay = [], [], []
    step = 0
    lane_ys = [float(ln.data.qpos[1]) for ln in lanes]
    for fi in range(n_frames):
        target = round(fi / FPS / DT)
        while step < target:
            if step == push_step:
                for ln in lanes:
                    ln.push(*push_vel)
            for ln, ly in zip(lanes, lane_ys):
                ln.step(race_cmd(ln.data.qpos, goal_x, ly) if step >= go_step else (0.0, 0.0, 0.0))
            step += 1
            if r0 <= step <= r1:
                replay.append([ln.qpos() for ln in lanes])
        frames.append([ln.qpos() for ln in lanes])
        fallen.append([ln.fallen for ln in lanes])
    return np.array(frames), np.array(fallen), np.array(replay)


# ---------- drawing ----------

class Stage:
    def __init__(self, model, width, height, max_geom=10000):
        import mujoco

        self.mj = mujoco
        self.m = model
        self.m.vis.global_.offwidth = max(self.m.vis.global_.offwidth, width)
        self.m.vis.global_.offheight = max(self.m.vis.global_.offheight, height)
        self.d = mujoco.MjData(model)
        self.r = mujoco.Renderer(model, height, width, max_geom=max_geom)
        self.w, self.h = width, height
        self.fovy = float(model.vis.global_.fovy)
        self.opt, self.pert = mujoco.MjvOption(), mujoco.MjvPerturb()
        mat = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, "groundplane")
        if mat >= 0:  # visual only: darker floor so the white ducks stand out
            model.mat_rgba[mat] = [0.55, 0.62, 0.60, 1.0]

    def draw(self, qposes, cam, lines=()):
        mj = self.mj
        c = mj.MjvCamera()
        c.type = mj.mjtCamera.mjCAMERA_FREE
        c.lookat[:] = cam["lookat"]
        c.distance, c.azimuth, c.elevation = cam["distance"], cam["azimuth"], cam["elevation"]
        self.d.qpos[:] = qposes[0]
        mj.mj_forward(self.m, self.d)
        self.r.update_scene(self.d, camera=c)
        scn = self.r.scene
        for q in qposes[1:]:
            self.d.qpos[:] = q
            mj.mj_forward(self.m, self.d)
            mj.mjv_addGeoms(self.m, self.d, self.opt, self.pert, mj.mjtCatBit.mjCAT_DYNAMIC, scn)
        for pos, size, rgba in lines:
            if scn.ngeom >= scn.maxgeom:
                break
            mj.mjv_initGeom(scn.geoms[scn.ngeom], mj.mjtGeom.mjGEOM_BOX, np.array(size, float),
                            np.array(pos, float), np.eye(3).flatten(), np.array(rgba, np.float32))
            scn.ngeom += 1
        return self.r.render()

    def close(self):
        self.r.close()


def course_lines(n_lanes, goal_x):
    y0, y1 = -SPACING / 2, (n_lanes - 0.5) * SPACING
    cy, hy = (y0 + y1) / 2, (y1 - y0) / 2
    lines = [((-0.12, cy, 0.001), (0.01, hy, 0.001), (1, 1, 1, 1)),
             ((goal_x, cy, 0.001), (0.015, hy, 0.001), (0.2, 0.75, 0.45, 1))]
    for i in range(n_lanes + 1):  # lane dividers
        y = (i - 0.5) * SPACING
        lines.append(((goal_x / 2 + 0.1, y, 0.001), (goal_x / 2 + 0.4, 0.004, 0.0008), (0.85, 0.88, 0.9, 1)))
    return lines


class Overlay:
    def __init__(self):
        from PIL import ImageFont

        self._fonts = {}
        self._ImageFont = ImageFont

    def font(self, size):
        if size not in self._fonts:
            self._fonts[size] = self._ImageFont.truetype(FONT, size)
        return self._fonts[size]

    def panel(self, draw, text, center, size, fg=(29, 36, 33), bg=(255, 255, 255, 235), pad=None):
        f = self.font(size)
        l, t, r, b = draw.textbbox((0, 0), text, font=f)
        pad = pad or size // 3
        w, h = r - l + 2 * pad, b - t + 2 * pad
        x, y = center[0] - w / 2, center[1] - h / 2
        draw.rounded_rectangle((x, y, x + w, y + h), radius=pad, fill=bg)
        draw.text((x + pad - l, y + pad - t), text, font=f, fill=fg)

    def shout(self, draw, text, center, size, fg=(226, 85, 27)):
        f = self.font(size)
        draw.text(center, text, font=f, fill=fg, anchor="mm", stroke_width=size // 10, stroke_fill=(255, 255, 255))

    def arrow(self, draw, p0, p1, color=(226, 85, 27), width=10):
        draw.line([p0, p1], fill=color, width=width)
        d = np.array(p1, float) - np.array(p0, float)
        d /= np.linalg.norm(d) + 1e-9
        n = np.array([-d[1], d[0]])
        tip = np.array(p1, float) + d * 22
        draw.polygon([tuple(tip), tuple(np.array(p1) + n * 18), tuple(np.array(p1) - n * 18)], fill=color)


def ease(a):
    a = min(max(a, 0.0), 1.0)
    return a * a * (3 - 2 * a)


def wide_cam(n_lanes, goal_x):
    # From behind the goal, looking back at the start: ducks run toward the viewer, lanes span the width.
    return dict(lookat=(goal_x / 2 + 0.1, (n_lanes - 1) * SPACING / 2, 0.05), distance=3.3, azimuth=180.0, elevation=-30.0)


def crowd_cam(a, focus):
    """a: 0 -> 1 over the crowd scene. Starts on one duck and pulls back over the whole crowd."""
    e = ease(a)
    return dict(lookat=(focus[0] * (1 - e), focus[1] * (1 - e), 0.12 * (1 - e)),
                distance=0.8 + (11.0 - 0.8) * e, azimuth=150.0 - 25.0 * a, elevation=-12.0 - 22.0 * e)


def end_cam(n_lanes, goal_x):
    return dict(lookat=(goal_x, (n_lanes - 1) * SPACING / 2, 0.0), distance=3.9, azimuth=180.0, elevation=-38.0)


def render(out_path, lanes_spec=None, push_time=None, push_vel=None, goal_x=None):
    import mediapy
    import imageio_ffmpeg
    from PIL import Image, ImageDraw

    from lanes import LANES, PUSH_TIME_S, PUSH_VEL, GOAL_X
    from race_sim import load_lanes

    lanes_spec = lanes_spec or LANES
    push_time = PUSH_TIME_S if push_time is None else push_time
    push_vel = PUSH_VEL if push_vel is None else push_vel
    goal_x = GOAL_X if goal_x is None else goal_x

    # Lane order left to right on screen: 0 minutes ... 3 hours.
    ordered = list(lanes_spec)
    lanes = load_lanes(ordered, spacing=SPACING)
    labels = [lb for _, lb in ordered]
    n = len(lanes)
    print("simulating...", flush=True)
    frames, fallen, replay = simulate(lanes, push_time, push_vel, goal_x)

    ys = frames[:, :, 1]
    min_gap = float(np.min(np.diff(ys, axis=1)))
    print(f"min adjacent lane gap over race: {min_gap:.3f} m", flush=True)
    assert min_gap > 0.25, "ducks drifted into a neighbouring lane"

    final = frames[-1]
    results = [result_label(bool(fallen[-1, i]), float(final[i, 0] - frames[0, i, 0]), goal_x) for i in range(n)]
    print("results:", list(zip(labels, results)), flush=True)

    print("simulating crowd...", flush=True)
    from crowd import simulate_crowd
    crowd_frames, _ = simulate_crowd(str(CROWD_CKPT), CROWD_SIDE, T_END - T_RESULTS_END, warmup=CROWD_WARMUP_S, seed=CROWD_SEED)
    centre = int(np.argmin(np.linalg.norm(crowd_frames[0, :, :2], axis=1)))
    crowd_focus = tuple(crowd_frames[0, centre, :2])
    stage = Stage(lanes[0].model, W, H)
    crowd_stage = Stage(lanes[0].model, W, H, max_geom=CROWD_SIDE * CROWD_SIDE * 60)
    half = Stage(lanes[0].model, W // 2, H)
    ov = Overlay()
    lines = course_lines(n, goal_x)
    wide = wide_cam(n, goal_x)
    out = []
    total = int(T_END * FPS)
    i_zero, i_best = 0, n - 1  # 0-minute lane and 3-hour lane
    for k in range(total):
        t = k / FPS
        if t < T_RACE_END:
            q = frames[min(int(round(t * FPS)), len(frames) - 1)]
            if t < T_PAN:
                cam = wide
            elif t < T_GO:
                a = ease((t - T_PAN) / (T_GO - T_PAN))
                cam = dict(lookat=(0.0, -0.1 + a * ((n - 1) * SPACING + 0.2), 0.12), distance=1.3, azimuth=125.0, elevation=-26.0)
            else:
                cam = wide
            img = Image.fromarray(stage.draw(q, cam, lines))
            dr = ImageDraw.Draw(img, "RGBA")
            show_labels = T_PAN <= t < T_GO or T_RACE <= t
            if show_labels:
                for i in range(n):
                    u, v = project(q[i, :3] + [0, 0, 0.30], cam, W, H, stage.fovy)
                    if 0 < u < W and 0 < v < H:
                        ov.panel(dr, labels[i], (u, v), LABEL_PX if t >= T_RACE else 48)
            if t < T_PAN:
                ov.panel(dr, TITLE, (W / 2, H / 2), 120)
            elif T_GO <= t < T_RACE:
                ov.shout(dr, GO_TEXT, (W / 2, H * 0.42), 150)
            push_t = T_GO + push_time
            if push_t - 0.3 <= t < push_t + 1.5:
                ov.shout(dr, PUSH_TEXT, (W / 2, H * 0.12), 110)
                for i in range(n):
                    p = q[i, :3] + [0, 0, 0.12]
                    a0 = project(p - [0, 0.40, 0], cam, W, H, stage.fovy)
                    a1 = project(p - [0, 0.16, 0], cam, W, H, stage.fovy)
                    ov.arrow(dr, a0, a1)
            frame = np.asarray(img)
        elif t < T_REPLAY_END:
            sim_t = REPLAY_SIM[0] + (t - T_RACE_END) * REPLAY_SPEED
            ri = min(int(round((sim_t - REPLAY_SIM[0]) / DT)), len(replay) - 1)
            q = replay[ri]
            halves = []
            for i, name in ((i_zero, labels[i_zero]), (i_best, labels[i_best])):
                start_y = i * SPACING
                cam = dict(lookat=(q[i, 0], start_y + 0.12, 0.14), distance=1.15, azimuth=180.0, elevation=-24.0)
                im = Image.fromarray(half.draw([q[i]], cam))
                d2 = ImageDraw.Draw(im, "RGBA")
                tw = d2.textlength(name, font=ov.font(56))
                ov.panel(d2, name, (40 + tw / 2 + 19, 90), 56)
                halves.append(np.asarray(im))
            img = Image.fromarray(np.concatenate(halves, axis=1))
            dr = ImageDraw.Draw(img, "RGBA")
            dr.rectangle((W // 2 - 4, 0, W // 2 + 4, H), fill=(255, 255, 255, 255))
            ov.panel(dr, "スロー", (W - 140, H - 80), 48, fg=(255, 255, 255), bg=(0, 0, 0, 150))
            frame = np.asarray(img)
        elif t < T_RESULTS_END:
            a = (t - T_REPLAY_END) / (T_RESULTS_END - T_REPLAY_END)
            cam = dict(end_cam(n, goal_x), azimuth=180.0 - 6.0 * a)
            img = Image.fromarray(stage.draw(final, cam, lines))
            dr = ImageDraw.Draw(img, "RGBA")
            for i in range(n):
                u, v = project(final[i, :3] + [0, 0, 0.30], cam, W, H, stage.fovy)
                ov.panel(dr, labels[i], (u, v - 34), LABEL_PX)
                ov.panel(dr, results[i], (u, v + 26), LABEL_PX, fg=(255, 255, 255), bg=(226, 85, 27, 235))
            if t >= T_REPLAY_END + 1.0:
                dr.rectangle((0, 0, W, H), fill=(0, 0, 0, 90))
                ov.panel(dr, END_LINES[0], (W / 2, H - 120), END_PX + 8)
            frame = np.asarray(img)
        else:
            a = (t - T_RESULTS_END) / (T_END - T_RESULTS_END)
            q = crowd_frames[min(int(round((t - T_RESULTS_END) * FPS)), len(crowd_frames) - 1)]
            cam = crowd_cam(min(1.0, a / 0.6), crowd_focus)
            img = Image.fromarray(crowd_stage.draw(q, cam))
            dr = ImageDraw.Draw(img, "RGBA")
            if t >= T_RESULTS_END + 2.5:
                ov.panel(dr, CROWD_TEXT, (W / 2, 110), END_PX + 8)
            if t >= T_RESULTS_END + 5.0:
                dr.rectangle((0, H - 300, W, H), fill=(0, 0, 0, 90))
                ov.panel(dr, END_LINES[1], (W / 2, H - 205), END_PX)
                ov.panel(dr, END_LINES[2], (W / 2, H - 95), END_PX)
            frame = np.asarray(img)
        fade = min(1.0, (k + 1) / (FADE_S * FPS), (total - k) / (FADE_S * FPS))
        if fade < 1.0:
            frame = (frame.astype(np.float32) * fade).astype(np.uint8)
        out.append(frame[:, :, :3])
        if k % 150 == 0:
            print(f"frame {k}/{total}", flush=True)
    stage.close()
    crowd_stage.close()
    half.close()
    mediapy.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
    mediapy.write_video(out_path, out, fps=FPS)
    print(f"wrote {out_path} ({len(out)} frames)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="race.mp4")
    render(ap.parse_args().out)

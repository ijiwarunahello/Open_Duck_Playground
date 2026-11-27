"""One duck per lane, each with its own MuJoCo model and ONNX policy.

The control loop matches playground/open_duck_mini_v2/mujoco_infer.py (MjInfer.run).
Run with cwd = repository root.
"""
from pathlib import Path

import numpy as np

from playground.open_duck_mini_v2.mujoco_infer import MjInfer, USE_MOTOR_SPEED_LIMITS

SCENE_XML = "playground/open_duck_mini_v2/xmls/scene_flat_terrain.xml"
REFERENCE = "playground/open_duck_mini_v2/data/polynomial_coefficients.pkl"
FALL_Z = 0.08  # base height below which a duck counts as fallen (standing ~0.16)


class Lane:
    def __init__(self, onnx_path: str, label: str, lane_y: float, lane_x: float = 0.0):
        self.label = label
        self.inf = MjInfer(SCENE_XML, REFERENCE, onnx_path, False)
        self.model, self.data = self.inf.model, self.inf.data
        self.data.qpos[0] += lane_x
        self.data.qpos[1] = lane_y
        self._start_qpos = self.data.qpos.copy()
        self.fallen = False
        self._counter = 0

    def reset(self) -> None:
        """Back to the start pose with zero velocity and a fresh action history
        (what a training episode reset does)."""
        import mujoco

        s, d = self.inf, self.data
        d.qpos[:] = self._start_qpos
        d.qvel[:] = 0.0
        s.last_action = np.zeros(s.num_dofs)
        s.last_last_action = np.zeros(s.num_dofs)
        s.last_last_last_action = np.zeros(s.num_dofs)
        s.motor_targets = s.default_actuator.copy()
        s.prev_motor_targets = s.default_actuator.copy()
        s.imitation_i = 0
        s.imitation_phase = np.array([0, 0])
        d.ctrl[:] = s.default_actuator
        mujoco.mj_forward(self.model, d)
        self.fallen = False
        self._counter = 0

    def qpos(self) -> np.ndarray:
        return self.data.qpos.copy()

    def push(self, vx: float, vy: float) -> None:
        self.data.qvel[0] += vx
        self.data.qvel[1] += vy

    def step(self, cmd) -> None:
        import mujoco

        s, d = self.inf, self.data
        s.commands[:3] = list(cmd)
        mujoco.mj_step(self.model, d)
        self._counter += 1
        if self._counter % s.decimation == 0:
            s.imitation_i = (s.imitation_i + s.phase_frequency_factor) % s.PRM.nb_steps_in_period
            ph = s.imitation_i / s.PRM.nb_steps_in_period * 2 * np.pi
            s.imitation_phase = np.array([np.cos(ph), np.sin(ph)])
            action = s.policy.infer(s.get_obs(d, s.commands))
            s.last_last_last_action = s.last_last_action.copy()
            s.last_last_action = s.last_action.copy()
            s.last_action = action.copy()
            s.motor_targets = s.default_actuator + action * s.action_scale
            if USE_MOTOR_SPEED_LIMITS:
                lim = s.max_motor_velocity * (s.sim_dt * s.decimation)
                s.motor_targets = np.clip(s.motor_targets, s.prev_motor_targets - lim, s.prev_motor_targets + lim)
                s.prev_motor_targets = s.motor_targets.copy()
            d.ctrl = s.motor_targets.copy()
        if d.qpos[2] < FALL_Z:
            self.fallen = True


def load_lanes(spec, spacing: float = 0.5):
    """spec: list of (onnx_path, label). Lane i starts at y = i * spacing."""
    missing = [p for p, _ in spec if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError(f"ONNX not found: {missing}")
    return [Lane(p, label, i * spacing) for i, (p, label) in enumerate(spec)]

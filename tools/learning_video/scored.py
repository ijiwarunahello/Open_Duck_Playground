"""A race Lane that also records what the policy saw, what it output and the
training reward terms (same functions and scales as playground/open_duck_mini_v2/joystick.py).

Run with cwd = repository root.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "race_video"))
from race_sim import Lane  # noqa: E402

from playground.common.rewards_numpy import (  # noqa: E402
    cost_action_rate,
    cost_stand_still,
    cost_torques,
    reward_alive,
    reward_tracking_ang_vel,
    reward_tracking_lin_vel,
)
from playground.open_duck_mini_v2.custom_rewards_numpy import reward_imitation  # noqa: E402

# joystick.py default_config().reward_config
SCALES = {
    "tracking_lin_vel": 2.5,
    "tracking_ang_vel": 6.0,
    "torques": -1.0e-3,
    "action_rate": -0.5,
    "stand_still": -0.2,
    "alive": 20.0,
    "imitation": 1.0,
}
TRACKING_SIGMA = 0.01
CTRL_DT = 0.02  # reward is sum(scaled terms) * dt per control step


def scaled_terms(raw: dict) -> dict:
    """Raw reward terms -> the per-step contribution used in training (scale * dt)."""
    return {k: float(v) * SCALES[k] * CTRL_DT for k, v in raw.items()}


def step_reward(raw: dict) -> float:
    """Training reward of one control step: clip(sum(scale * term) * dt, 0, 10000)."""
    return float(np.clip(sum(scaled_terms(raw).values()), 0.0, 10000.0))


class ScoredLane(Lane):
    """Lane.step plus, at every policy call, `last` = dict(obs, action, raw rewards)."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.last = None
        self._prev_action = np.zeros(self.inf.num_dofs)

    def step(self, cmd) -> None:
        s, d = self.inf, self.data
        policy_tick = (self._counter + 1) % s.decimation == 0
        super().step(cmd)
        if not policy_tick:
            return
        cmd3 = np.asarray(s.commands, float)
        action = s.last_action
        obs = s.get_obs(d, s.commands)  # same vector the policy just used (one physics step later)
        contacts = np.array(s.get_feet_contacts(d), dtype=float)
        ref = s.PRM.get_reference_motion(*cmd3[:3], s.imitation_i)
        jq = s.get_actuator_joints_qpos(d.qpos)
        jv = s.get_actuator_joints_qvel(d.qvel)
        raw = {
            "tracking_lin_vel": reward_tracking_lin_vel(cmd3, s.get_linvel(d), TRACKING_SIGMA),
            "tracking_ang_vel": reward_tracking_ang_vel(cmd3, s.get_gyro(d), TRACKING_SIGMA),
            "torques": cost_torques(d.actuator_force),
            "action_rate": cost_action_rate(action, self._prev_action),
            "alive": reward_alive(),
            "imitation": reward_imitation(
                s.get_floating_base_qpos(d.qpos), s.get_floating_base_qvel(d.qvel),
                jq, jv, contacts, ref, cmd3, True,
            ),
            "stand_still": cost_stand_still(cmd3, jq, jv, s.default_actuator, ignore_head=False),
        }
        self._prev_action = action.copy()
        self.last = {"obs": obs.copy(), "action": action.copy(), "raw": raw,
                     "gravity": np.asarray(s.get_gravity(d), float).copy(), "ref": ref}


# On-screen cards (scene 3): training reward terms grouped into four rules.
CARDS = (
    ("alive", ("alive",)),
    ("follow", ("tracking_lin_vel", "tracking_ang_vel")),
    ("imitate", ("imitation",)),
    ("jitter", ("action_rate", "torques")),
)
# Full bar = the largest value the "+" terms can reach per control step (scale * term max):
# alive 20 * 1; follow 2.5 * 1 + 6 * 1; imitation 1 * (1 + 1 + 0.5 + 0.5 + 2 contacts) (joint terms are <= 0).
CARD_FULL = {"alive": 20.0, "follow": 8.5, "imitate": 5.0}


def card_values(raw: dict) -> dict:
    """Per-step scaled reward per card (no dt), jitter as a positive magnitude of the cost."""
    out = {}
    for name, keys in CARDS:
        v = sum(float(raw[k]) * SCALES[k] for k in keys)
        out[name] = -v if name == "jitter" else v
    return out

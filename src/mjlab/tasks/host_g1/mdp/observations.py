# src/mjlab/tasks/standup/mdp/observations.py
"""HoST observation terms — mjlab port.

Source: legged_gym/envs/g1/g1_config_ground.py::env
  num_one_step_observations = 76
  num_actor_history = 6  (handled by ObsGroup history_length in env cfg, not here)

Per-step observation = projected_gravity(3) + base_ang_vel(3) + joint_pos_rel(23)
                      + joint_vel_rel(23) + last_action(23) + extras(~1) = 76
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def projected_gravity(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Gravity vector in root body frame. IMU-equivalent signal."""
    asset: Entity = env.scene[asset_cfg.name]
    gravity_w = asset.data.gravity_vec_w
    return quat_apply_inverse(asset.data.root_link_quat_w, gravity_w)


def base_ang_vel(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Root angular velocity in the root body frame."""
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.root_link_ang_vel_b


def base_lin_vel(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Root linear velocity in the root body frame. Privileged-obs only in HoST
    (not estimable from onboard IMU); exclude from the policy obs group."""
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.root_link_lin_vel_b


def base_height(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Root height in world frame. Privileged-obs only (needs terrain height
    in general, but flat ground makes this equal to root z)."""
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.root_link_pos_w[:, 2].unsqueeze(-1)


def joint_pos_rel(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Joint positions relative to default pose."""
    asset: Entity = env.scene[asset_cfg.name]
    pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    default = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return pos - default


def joint_vel_rel(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Joint velocities."""
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.joint_vel[:, asset_cfg.joint_ids]


def last_action(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Previous policy action (matches IsaacLab's mdp.last_action)."""
    return env.action_manager.action


def feet_contact_state(
    env: ManagerBasedRlEnv,
    sensor_name: str = "feet_contact_sensor",
) -> torch.Tensor:
    """Binary foot contact flags. Not in original 76-dim vector but useful
    for curriculum gating; add only if num_one_step_observations is bumped."""
    sensor = env.scene[sensor_name]
    found = sensor.data.found
    assert found is not None
    return (found > 0).float()

# projected_gravity  (3)
# base_ang_vel       (3)
# joint_pos_rel      (23)
# joint_vel_rel      (23)
# prev_action        (23)
# extra_feature      (1)
# ---------------------
# total              76
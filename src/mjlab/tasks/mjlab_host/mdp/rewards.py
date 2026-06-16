# src/mjlab/tasks/standup/mdp/rewards.py
"""
HoST reward terms — mjlab port.

Derived from:
  legged_gym/envs/g1/g1_config_ground.py       (scales, groups, sigmas)
  legged_gym/envs/g1/g1_config_ground_prone.py
  legged_gym/envs/g1/g1_config_slope.py
  legged_gym/envs/g1/g1_config_wall.py
  legged_gym/envs/g1/g1_config_platform.py

Each function is a reward term registered with mjlab's RewardManager.
Signature contract:
    func(env: ManagerBasedRlEnv, **params) -> Tensor[num_envs]

The RewardManager multiplies the return value by `weight * env.step_dt`
automatically, so every function returns the *raw* unscaled signal.

HoST uses four reward groups with separate critic heads:
  GROUP 1 — task:   orientation + head height
  GROUP 2 — regu:   torques, vel, acc, action rate, smoothness, joint limits
  GROUP 3 — style:  posture deviation, foot placement, shank orientation, etc.
  GROUP 4 — target: post-standup stability (ang/lin vel, upper-body pose, etc.)

Group weights from g1_config_ground.py:
  reward_group_weights = [2.5, 0.1, 1.0, 1.0]
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _projected_gravity(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> Tensor:
    """Gravity vector [0,0,-1] projected into the robot body frame. Shape: [N, 3]."""
    robot: Entity = env.scene[asset_cfg.name]
    gravity_world = torch.zeros(env.num_envs, 3, device=env.device)
    gravity_world[:, 2] = -1.0
    root_quat = getattr(robot.data, "root_quat_w")
    return quat_apply_inverse(root_quat, gravity_world)


def _gaussian(x: Tensor, sigma: float) -> Tensor:
    """Gaussian kernel:  exp(-x^2 / (2 * sigma^2))."""
    return torch.exp(-x / (2.0 * sigma ** 2))


# ─────────────────────────────────────────────────────────────────────────────
# GROUP 1 — Task rewards
# weight in cfg: task_orientation=1, task_head_height=1
# group weight: 2.5  (g1_config_ground / platform / wall / slope)
#               1.0  (g1_config_ground_prone)
# ─────────────────────────────────────────────────────────────────────────────

def task_orientation(
    env: ManagerBasedRlEnv,
    sigma: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> Tensor:
    """
    Reward the robot for being upright.

    When upright, projected_gravity ≈ [0, 0, -1].
    We measure the squared lateral deviation (x² + y²) and pass it through
    a Gaussian kernel so the reward is 1.0 when perfectly upright and decays
    smoothly when tilted.

    Ports:
      _reward_orientation (host_ground.py) &
      task_orientation scale from g1_config_ground.py::rewards.scales

    sigma=1.0 matches `orientation_sigma = 1` in all g1 configs.
    """
    grav_b = _projected_gravity(env, asset_cfg)             # [N, 3]
    lateral_sq = grav_b[:, 0] ** 2 + grav_b[:, 1] ** 2    # 0 when upright
    return _gaussian(lateral_sq, sigma)


def task_head_height(
    env: ManagerBasedRlEnv,
    target_height: float = 1.0,
    margin: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> Tensor:
    """
    Reward for raising the head (approximated by root/pelvis height here;
    the IsaacGym version uses a keyframe body called 'keyframe_head').

    Uses a Gaussian kernel over the height error so the reward rises
    smoothly as the robot stands up.

    Ports:
      _reward_head_height / task_head_height scale from g1_config_ground.py

    target_height=1.0 and margin=1.0 from:
      rewards.target_head_height = 1
      rewards.target_head_margin = 1
    """
    robot: Entity = env.scene[asset_cfg.name]
    root_pos = getattr(robot.data, "root_pos_w")
    head_z = root_pos[:, 2]          # pelvis height as proxy
    height_err_sq = (head_z - target_height) ** 2
    return _gaussian(height_err_sq, margin)


# ─────────────────────────────────────────────────────────────────────────────
# GROUP 2 — Regularisation penalties  (all weights are negative in source)
# group weight: 0.1  (all g1 configs)
# ─────────────────────────────────────────────────────────────────────────────

def regu_dof_acc(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> Tensor:
    """
    Penalise large joint accelerations:  ||ddq||²

    Scale from all g1 configs: regu_dof_acc = -2.5e-7
    Ports `_reward_dof_acc` from host_ground.py.
    """
    robot: Entity = env.scene[asset_cfg.name]
    return torch.sum(robot.data.joint_acc ** 2, dim=-1)


def regu_action_rate(
    env: ManagerBasedRlEnv,
) -> Tensor:
    """
    Penalise large action changes between consecutive steps:  ||a_t - a_{t-1}||²

    Scale from all g1 configs: regu_action_rate = -0.01
    Ports `_reward_action_rate` from host_ground.py.
    """
    return torch.sum(
        (env.action_manager.action - env.action_manager.prev_action) ** 2,
        dim=-1,
    )


def regu_smoothness(
    env: ManagerBasedRlEnv,
) -> Tensor:
    """
    Second-order smoothness (jerk) penalty:  ||Δa_t - Δa_{t-1}||²

    This is HoST's key contribution for preventing oscillatory / violent
    motions on hardware — the "implicit motion speed bound".

    Scale from all g1 configs: regu_smoothness = -0.01
    Ports the smoothness term from host_ground.py.

    Requires action manager to expose `prev_prev_action` (two steps ago).
    If your mjlab version only has `prev_action`, this falls back to the
    same signal as regu_action_rate; register a custom action manager that
    stores an extra history step to get the full jerk term.
    """
    a  = env.action_manager.action
    a1 = env.action_manager.prev_action
    # attempt to get a_{t-2}; fall back gracefully
    a2 = getattr(env.action_manager, "prev_prev_action", a1)
    delta_t   = a  - a1
    delta_t_1 = a1 - a2
    return torch.sum((delta_t - delta_t_1) ** 2, dim=-1)


def regu_torques(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> Tensor:
    """
    Penalise large joint torques:  ||τ||²

    Scale from all g1 configs: regu_torques = -2.5e-6
    Ports `_reward_torques` from host_ground.py.
    """
    robot: Entity = env.scene[asset_cfg.name]
    return torch.sum(robot.data.joint_torques ** 2, dim=-1)


def regu_joint_power(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> Tensor:
    """
    Penalise mechanical power consumed:  ||τ · dq||

    Scale from all g1 configs: regu_joint_power = -2.5e-5
    """
    robot: Entity = env.scene[asset_cfg.name]
    return torch.sum(
        torch.abs(robot.data.joint_torques * robot.data.joint_vel),
        dim=-1,
    )


def regu_dof_vel(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> Tensor:
    """
    Penalise large joint velocities:  ||dq||²

    Scale from all g1 configs: regu_dof_vel = -1e-3
    Ports `_reward_dof_vel` from host_ground.py.
    """
    robot: Entity = env.scene[asset_cfg.name]
    return torch.sum(robot.data.joint_vel ** 2, dim=-1)


def regu_joint_tracking_error(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> Tensor:
    """
    Penalise deviation of joint positions from their default (rest) pose.

    Scale from all g1 configs: regu_joint_tracking_error = -0.00025
    """
    robot: Entity = env.scene[asset_cfg.name]
    default_pos = robot.data.default_joint_pos          # [N, num_joints]
    error = robot.data.joint_pos - default_pos
    return torch.sum(error ** 2, dim=-1)


def regu_dof_pos_limits(
    env: ManagerBasedRlEnv,
    soft_ratio: float = 0.9,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> Tensor:
    """
    Penalise joint positions that exceed soft limits (90 % of hard limits).

    Scale from all g1 configs: regu_dof_pos_limits = -100.0
    soft_dof_pos_limit = 0.9 from g1_config_ground.py::rewards
    """
    robot: Entity = env.scene[asset_cfg.name]
    pos        = robot.data.joint_pos                    # [N, J]
    pos_limits = robot.data.joint_pos_limits             # [J, 2] — (lower, upper)
    lower = pos_limits[:, 0] * soft_ratio
    upper = pos_limits[:, 1] * soft_ratio
    out_of_bounds = (
        torch.clamp(lower - pos, min=0.0) +
        torch.clamp(pos - upper, min=0.0)
    )
    return torch.sum(out_of_bounds ** 2, dim=-1)


def regu_dof_vel_limits(
    env: ManagerBasedRlEnv,
    soft_ratio: float = 0.9,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> Tensor:
    """
    Penalise joint velocities that exceed soft velocity limits.

    Scale from all g1 configs: regu_dof_vel_limits = -1.0
    soft_dof_vel_limit = 0.9 from g1_config_ground.py::rewards
    """
    robot: Entity = env.scene[asset_cfg.name]
    vel        = robot.data.joint_vel                    # [N, J]
    vel_limits = robot.data.joint_vel_limits             # [J]  (symmetric)
    soft_limit = vel_limits * soft_ratio
    excess = torch.clamp(torch.abs(vel) - soft_limit, min=0.0)
    return torch.sum(excess ** 2, dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# GROUP 3 — Style rewards  (mix of positive + negative weights)
# group weight: 1.0  (all g1 configs)
# ─────────────────────────────────────────────────────────────────────────────

def style_waist_deviation(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> Tensor:
    """
    Penalise yaw deviation of the waist joint from zero.

    Scale: style_waist_deviation = -10
    Joint: waist_yaw_joint (index resolved via asset_cfg joint_names).
    """
    robot: Entity = env.scene[asset_cfg.name]
    # find waist_yaw_joint index at runtime via resolved names
    joint_ids = asset_cfg.joint_ids if hasattr(asset_cfg, "joint_ids") else None
    if joint_ids is not None:
        waist_pos = robot.data.joint_pos[:, joint_ids]
    else:
        # fallback: use all waist joints filtered by name at env init
        waist_pos = robot.data.joint_pos[:, asset_cfg.joint_indices]
    return torch.sum(waist_pos ** 2, dim=-1)


def style_hip_yaw_deviation(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> Tensor:
    """
    Penalise hip yaw deviation from zero.

    Scale: style_hip_yaw_deviation = -10
    Joints: left_hip_yaw_joint, right_hip_yaw_joint
    """
    robot: Entity = env.scene[asset_cfg.name]
    hip_yaw = robot.data.joint_pos[:, asset_cfg.joint_indices]
    return torch.sum(hip_yaw ** 2, dim=-1)


def style_hip_roll_deviation(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> Tensor:
    """
    Penalise hip roll deviation from zero.

    Scale: style_hip_roll_deviation = -10
    Joints: left_hip_roll_joint, right_hip_roll_joint
    """
    robot: Entity = env.scene[asset_cfg.name]
    hip_roll = robot.data.joint_pos[:, asset_cfg.joint_indices]
    return torch.sum(hip_roll ** 2, dim=-1)


def style_shoulder_roll_deviation(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> Tensor:
    """
    Penalise shoulder roll deviation from the default arm pose.

    Scale: style_shoulder_roll_deviation = -2.5
    Joints: left_shoulder_roll_joint, right_shoulder_roll_joint
    """
    robot: Entity = env.scene[asset_cfg.name]
    pos     = robot.data.joint_pos[:, asset_cfg.joint_indices]
    default = robot.data.default_joint_pos[:, asset_cfg.joint_indices]
    return torch.sum((pos - default) ** 2, dim=-1)


def style_knee_deviation(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> Tensor:
    """
    Penalise knee joint deviation.

    Scale (ground): style_knee_deviation = -0.25
    Scale (platform / slope / wall): style_knee_deviation = -10
    Joints: left_knee_joint, right_knee_joint
    """
    robot: Entity = env.scene[asset_cfg.name]
    pos     = robot.data.joint_pos[:, asset_cfg.joint_indices]
    default = robot.data.default_joint_pos[:, asset_cfg.joint_indices]
    return torch.sum((pos - default) ** 2, dim=-1)


def style_feet_distance(
    env: ManagerBasedRlEnv,
    target_distance: float = 0.25,
    sigma: float = 2.0,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_sensor"),
) -> Tensor:
    """
    Penalise feet that are too close together or too far apart.

    Scale: style_feet_distance = -10
    Uses foot body positions from the contact sensor or robot entity.
    """
    robot: Entity = env.scene["robot"]
    # body positions for left/right ankle roll links
    left_ids  = sensor_cfg.body_ids[0:1]
    right_ids = sensor_cfg.body_ids[1:2]
    left_pos  = robot.data.body_pos_w[:, left_ids,  :]   # [N, 1, 3]
    right_pos = robot.data.body_pos_w[:, right_ids, :]   # [N, 1, 3]
    dist = torch.norm(left_pos[:, 0, :2] - right_pos[:, 0, :2], dim=-1)   # [N]
    err  = (dist - target_distance) ** 2
    return _gaussian(err, sigma)


def style_shank_orientation(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> Tensor:
    """
    Reward for the shank (lower leg) being roughly vertical.
    Approximated here via ankle joint positions being near default.

    Scale (ground / wall): style_shank_orientation = 10  (positive — reward)
    Joints: left_ankle_pitch_joint, right_ankle_pitch_joint
    """
    robot: Entity = env.scene[asset_cfg.name]
    pos     = robot.data.joint_pos[:, asset_cfg.joint_indices]
    default = robot.data.default_joint_pos[:, asset_cfg.joint_indices]
    err_sq  = torch.sum((pos - default) ** 2, dim=-1)
    return torch.exp(-err_sq)


def style_ground_parallel(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> Tensor:
    """
    Reward for the foot being parallel to the ground (ankle roll ≈ 0).

    Scale (ground / wall): style_ground_parallel = 20  (positive — reward)
    Joints: left_ankle_roll_joint, right_ankle_roll_joint
    """
    robot: Entity = env.scene[asset_cfg.name]
    ankle_roll = robot.data.joint_pos[:, asset_cfg.joint_indices]
    return torch.exp(-torch.sum(ankle_roll ** 2, dim=-1))


def style_ang_vel_xy(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> Tensor:
    """
    Reward for keeping body angular velocity in XY small (stable standup).

    Scale (ground): style_style_ang_vel_xy = 1
    Scale (prone):  style_style_ang_vel_xy = 25
    """
    robot: Entity = env.scene[asset_cfg.name]
    ang_vel_xy = robot.data.root_ang_vel_b[:, :2]     # [N, 2]
    return torch.exp(-torch.sum(ang_vel_xy ** 2, dim=-1))


def style_left_foot_displacement(
    env: ManagerBasedRlEnv,
    sigma: float = 2.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> Tensor:
    """
    Reward for left foot staying near its default XY displacement from pelvis.

    Scale: style_left_foot_displacement = 2.5  (positive — reward)
    left_foot_displacement_sigma = -2 (used as sigma magnitude in source)
    """
    robot: Entity = env.scene[asset_cfg.name]
    pelvis_pos = robot.data.root_pos_w[:, :2]          # [N, 2]
    # left ankle roll body position
    left_foot_pos = robot.data.body_pos_w[:, asset_cfg.body_ids[0], :2]   # [N, 2]
    disp_sq = torch.sum((left_foot_pos - pelvis_pos) ** 2, dim=-1)
    return _gaussian(disp_sq, sigma)


def style_right_foot_displacement(
    env: ManagerBasedRlEnv,
    sigma: float = 2.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> Tensor:
    """
    Reward for right foot staying near its default XY displacement from pelvis.

    Scale: style_right_foot_displacement = 2.5  (positive — reward)
    """
    robot: Entity = env.scene[asset_cfg.name]
    pelvis_pos = robot.data.root_pos_w[:, :2]
    right_foot_pos = robot.data.body_pos_w[:, asset_cfg.body_ids[0], :2]
    disp_sq = torch.sum((right_foot_pos - pelvis_pos) ** 2, dim=-1)
    return _gaussian(disp_sq, sigma)


def style_feet_stumble(
    env: ManagerBasedRlEnv,
    threshold: float = 0.1,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_sensor"),
) -> Tensor:
    """
    Penalise feet making contact with nearly-vertical surfaces (stumbling).

    Scale (platform / slope / wall): style_feet_stumble = -25
    Returns count of stumble contacts per env.
    """
    contact_sensor: ContactSensor = env.scene[sensor_cfg.name]
    # net_forces_w: [N, num_bodies, 3]
    forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]  # [N, 2, 3]
    # Large horizontal contact force relative to vertical → stumble
    horiz = torch.norm(forces[:, :, :2], dim=-1)   # [N, 2]
    vert  = torch.abs(forces[:, :, 2])              # [N, 2]
    stumble = (horiz > threshold) & (horiz > vert)
    return stumble.float().sum(dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# GROUP 4 — Target / post-task rewards  (all positive weights)
# Activated only after the robot has nearly stood up (curriculum phase 3)
# group weight: 1.0  (all g1 configs)
# ─────────────────────────────────────────────────────────────────────────────

def target_ang_vel_xy(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> Tensor:
    """
    Reward zero XY angular velocity once the robot is standing.

    Scale: target_ang_vel_xy = 10
    """
    robot: Entity = env.scene[asset_cfg.name]
    ang_vel_xy = robot.data.root_ang_vel_b[:, :2]
    return torch.exp(-torch.sum(ang_vel_xy ** 2, dim=-1))


def target_lin_vel_xy(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> Tensor:
    """
    Reward zero XY linear velocity once the robot is standing (stay still).

    Scale: target_lin_vel_xy = 10
    """
    robot: Entity = env.scene[asset_cfg.name]
    lin_vel_xy = robot.data.root_lin_vel_b[:, :2]
    return torch.exp(-torch.sum(lin_vel_xy ** 2, dim=-1))


def target_feet_height_var(
    env: ManagerBasedRlEnv,
    sigma: float = 0.05,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_sensor"),
) -> Tensor:
    """
    Reward for both feet being at roughly the same height (level stance).

    Scale: target_feet_height_var = 2.5
    """
    robot: Entity = env.scene["robot"]
    left_z  = robot.data.body_pos_w[:, sensor_cfg.body_ids[0], 2]
    right_z = robot.data.body_pos_w[:, sensor_cfg.body_ids[1], 2]
    height_var = (left_z - right_z) ** 2
    return _gaussian(height_var, sigma)


def target_upper_dof_pos(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> Tensor:
    """
    Reward upper body joints (shoulders, elbows, wrists) being near the
    target standing pose defined in init_state.target_joint_angles.

    Scale: target_target_upper_dof_pos = 10
    Joints (from g1_config_ground.py::asset.left_arm_joints + right_arm_joints):
      left/right: shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll
    """
    robot: Entity = env.scene[asset_cfg.name]
    pos     = robot.data.joint_pos[:, asset_cfg.joint_indices]
    # target_joint_angles from g1_config_ground.py init_state
    target  = robot.data.default_joint_pos[:, asset_cfg.joint_indices]
    err_sq  = torch.sum((pos - target) ** 2, dim=-1)
    return torch.exp(-err_sq)


def target_orientation(
    env: ManagerBasedRlEnv,
    sigma: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> Tensor:
    """
    Reward for upright orientation in the post-standup phase.
    Same kernel as task_orientation but registered separately so it can
    be weighted by the target critic head.

    Scale: target_target_orientation = 10
    """
    grav_b = _projected_gravity(env, asset_cfg)
    lateral_sq = grav_b[:, 0] ** 2 + grav_b[:, 1] ** 2
    return _gaussian(lateral_sq, sigma)


def target_base_height(
    env: ManagerBasedRlEnv,
    target_height: float = 0.75,
    sigma: float = 0.25,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> Tensor:
    """
    Reward for maintaining the target standing base height.

    Scale: target_target_base_height = 10
    target_height = base_height_target = 0.75  (g1_config_ground.py)
    sigma         = base_height_sigma  = 0.25  (g1_config_ground_prone.py,
                                                 g1_config_slope.py,
                                                 g1_config_wall.py)
    """
    robot: Entity = env.scene[asset_cfg.name]
    root_z    = robot.data.root_pos_w[:, 2]
    height_err_sq = (root_z - target_height) ** 2
    return _gaussian(height_err_sq, sigma)
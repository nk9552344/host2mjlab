import torch
import numpy as np
from typing import Tuple, Dict

# Fixed framework import paths based on the source structure
from mjlab.tasks.base import BaseEnv
from mjlab.tasks.standup.standup_env_cfg import StandUpEnvCfg
from mjlab.tasks.standup.utils.standup_utils import quat_conjugate, quat_apply

def quat_rotate_inverse(q, v):
    q_conj = quat_conjugate(q)
    return quat_apply(q_conj, v)

class StandUpEnv(BaseEnv):
    def __init__(self, cfg: StandUpEnvCfg, sim_device: str, headless: bool):
        self.cfg = cfg
        self.device = sim_device
        self.headless = headless
        
        super().__init__(self.cfg, self.device, self.headless)
        
        self.num_envs = self.cfg.env.num_envs
        self.num_obs = self.cfg.env.num_observations
        self.num_actions = self.cfg.env.num_actions
        self.num_one_step_obs = self.cfg.env.num_one_step_observations
        self.num_history = self.cfg.env.num_actor_history
        
        self.obs_history_buf = torch.zeros(self.num_envs, self.num_obs, device=self.device, dtype=torch.float)
        self.actions = torch.zeros(self.num_envs, self.num_actions, device=self.device, dtype=torch.float)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, device=self.device, dtype=torch.float)
        
        self.default_dof_pos = torch.zeros(self.num_actions, device=self.device, dtype=torch.float)
        for i, name in enumerate(self.dof_names):
            if name in self.cfg.init_state.default_joint_angles:
                self.default_dof_pos[i] = self.cfg.init_state.default_joint_angles[name]
                
        self.p_gains = torch.zeros(self.num_actions, device=self.device, dtype=torch.float)
        self.d_gains = torch.zeros(self.num_actions, device=self.device, dtype=torch.float)
        for i, name in enumerate(self.dof_names):
            for key in self.cfg.control.stiffness.keys():
                if key in name:
                    self.p_gains[i] = self.cfg.control.stiffness[key]
                    self.d_gains[i] = self.cfg.control.damping[key]

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
            
        super().reset_idx(env_ids)
        
        self.obs_history_buf[env_ids] = 0.0
        self.actions[env_ids] = 0.0
        self.last_actions[env_ids] = 0.0
        
        init_pos = torch.tensor(self.cfg.init_state.pos, device=self.device)
        init_rot = torch.tensor(self.cfg.init_state.rot, device=self.device)
        
        self.root_pos[env_ids] = init_pos
        self.root_quat[env_ids] = init_rot
        self.root_lin_vel[env_ids] = 0.0
        self.root_ang_vel[env_ids] = 0.0
        
        self.dof_pos[env_ids] = self.default_dof_pos
        self.dof_vel[env_ids] = 0.0
        
        self.set_sim_state_idx(env_ids)
        self.compute_observations()

    def step(self, actions) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        self.last_actions[:] = self.actions[:]
        self.actions[:] = torch.clamp(actions, -self.cfg.control.clip_actions, self.cfg.control.clip_actions, out=self.actions)
        
        scaled_actions = self.actions * self.cfg.control.action_scale
        torque_targets = scaled_actions + self.default_dof_pos
        
        # Accessing nested sim subclass explicitly
        for _ in range(self.cfg.sim.decimation):
            torques = self.p_gains * (torque_targets - self.dof_pos) - self.d_gains * self.dof_vel
            self.apply_torques(torques)
            self.sim_substep()
            
        self.post_physics_checks()
        self.compute_observations()
        self.compute_rewards()
        
        return self.obs_history_buf, self.rew_buf, self.reset_buf, self.extras

    def compute_observations(self):
        gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=self.device).repeat(self.num_envs, 1)
        projected_gravity = quat_rotate_inverse(self.root_quat, gravity_vec)
        
        body_ang_vel = quat_rotate_inverse(self.root_quat, self.root_ang_vel)
        body_lin_vel = quat_rotate_inverse(self.root_quat, self.root_lin_vel)
        
        dof_pos_scaled = self.dof_pos - self.default_dof_pos
        dof_vel_scaled = self.dof_vel
        
        current_obs = torch.cat([
            body_ang_vel,
            body_lin_vel,
            projected_gravity,
            dof_pos_scaled,
            dof_vel_scaled,
            self.actions
        ], dim=-1)
        
        self.obs_history_buf = torch.cat([
            self.obs_history_buf[:, self.num_one_step_obs:],
            current_obs
        ], dim=-1)

    def compute_rewards(self):
        self.rew_buf[:] = 0.0
        
        # Enforcing explicit boolean conditions on PyTorch tensors to fix torch.where
        cond1 = self.progress_buf < 100
        cond2 = self.progress_buf < 300
        
        target_height = torch.where(
            cond1,
            torch.full_like(self.progress_buf, self.cfg.rewards.target_base_height_phase1, dtype=torch.float),
            torch.where(
                cond2,
                torch.full_like(self.progress_buf, self.cfg.rewards.target_base_height_phase2, dtype=torch.float),
                torch.full_like(self.progress_buf, self.cfg.rewards.target_base_height_phase3, dtype=torch.float)
            )
        )
        
        height_error = torch.square(self.root_pos[:, 2] - target_height)
        r_task = torch.exp(-height_error / self.cfg.rewards.tracking_sigma)
        
        up_vector = torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(self.num_envs, 1)
        projected_up = quat_rotate_inverse(self.root_quat, up_vector)
        orientation_error = torch.square(projected_up[:, 2] - 1.0)
        r_style = torch.exp(-orientation_error / self.cfg.rewards.tracking_sigma_orn)
        
        r_regu = -torch.sum(torch.square(self.actions), dim=-1)
        r_target = -torch.sum(torch.square(self.dof_pos - self.default_dof_pos), dim=-1)
        
        w_task, w_regu, w_style, w_target = self.cfg.rewards.reward_group_weights
        
        self.rew_buf += w_task * r_task + w_regu * r_regu + w_style * r_style + w_target * r_target

    def post_physics_checks(self):
        self.progress_buf += 1
        
        # Enforcing explicit boolean condition for the rollout length timeout
        max_steps = int(self.cfg.env.episode_length_s / self.cfg.sim.dt / self.cfg.sim.decimation)
        timeout_cond = self.progress_buf >= max_steps
        
        self.reset_buf[:] = torch.where(
            timeout_cond,
            torch.ones_like(self.reset_buf),
            torch.zeros_like(self.reset_buf)
        )
        
        if self.cfg.termination.penalize_illegal_contacts:
            contact_forces = self.get_contact_forces()
            for body_name in self.cfg.rewards.penalize_contacts_on:
                if body_name in self.body_names:
                    body_idx = self.body_names.index(body_name)
                    forces = torch.norm(contact_forces[:, body_idx, :], dim=-1)
                    illegal_contact = forces > self.cfg.rewards.contact_force_threshold
                    self.reset_buf = torch.where(illegal_contact, torch.ones_like(self.reset_buf), self.reset_buf)
                    
        low_height = self.root_pos[:, 2] < self.cfg.termination.terminal_body_height
        self.reset_buf = torch.where(low_height, torch.ones_like(self.reset_buf), self.reset_buf)
        
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            self.reset_idx(env_ids)
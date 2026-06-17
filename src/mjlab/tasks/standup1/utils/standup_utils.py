import os
import random
import pickle
import torch
import numpy as np

def quat_conjugate(q):
    shape = q.shape
    q_conj = q.clone()
    q_conj[..., 1:] = -q[..., 1:]
    return q_conj

def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    
    return torch.stack([w, x, y, z], dim=-1)

def quat_apply(q, v):
    shape = v.shape
    q_v = torch.zeros(shape[:-1] + (4,), dtype=v.dtype, device=v.device)
    q_v[..., 1:] = v
    q_conj = quat_conjugate(q)
    return quat_mul(quat_mul(q, q_v), q_conj)[..., 1:]

def extract_yaw(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return yaw

def quat_from_yaw(yaw):
    w = torch.cos(yaw * 0.5)
    z = torch.sin(yaw * 0.5)
    zeros = torch.zeros_like(w)
    return torch.stack([w, zeros, zeros, z], dim=-1)

def quat_apply_yaw_inverse(q, v):
    yaw = extract_yaw(q)
    q_yaw_inv = quat_from_yaw(-yaw)
    return quat_apply(q_yaw_inv, v)

def quat_mul_yaw_inverse(q1, q2):
    yaw = extract_yaw(q1)
    q_yaw_inv = quat_from_yaw(-yaw)
    return quat_mul(q_yaw_inv, q2)

def load_imitation_dataset(folder, mapping="joint_id.txt", suffix=".npz"):
    if not os.path.exists(folder):
        return [], {}
    filenames = [name for name in os.listdir(folder) if name.endswith(suffix)]
    dataset = {}
    for filename in filenames:
        try:
            with open(os.path.join(folder, filename), 'rb') as f:
                data = pickle.load(f)
                dataset[filename[:-len(suffix)]] = data
        except Exception:
            continue
    dataset_list = list(dataset.values())
    random.shuffle(dataset_list)
    
    joint_id_dict = {}
    if os.path.exists(mapping):
        with open(mapping, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    joint_id_dict[parts[1]] = int(parts[0])
    return dataset_list, joint_id_dict

class MotionLib:
    def __init__(self, datasets, mapping, dof_names, keyframe_names, fps=30, min_dt=0.1, device="cpu"):
        self.datasets = datasets
        self.mapping = mapping
        self.dof_names = dof_names
        self.keyframe_names = keyframe_names
        self.fps = fps
        self.min_dt = min_dt
        self.device = device
        
        self.lengths = [data['pos'].shape[0] for data in self.datasets]
        self.total_frames = sum(self.lengths)
        
    def sample_keyframes(self, batch_size):
        motion_ids = torch.randint(0, len(self.datasets), (batch_size,), device=self.device)
        keyframes = {}
        
        pos_list, quat_list, lin_vel_list, ang_vel_list, dof_list = [], [], [], [], []
        
        for i in range(batch_size):
            m_id = int(motion_ids[i].item())
            data = self.datasets[m_id]
            max_frame = int(self.lengths[m_id])
            frame_idx = random.randint(0, max_frame - 1)
            
            pos_list.append(torch.tensor(data['pos'][frame_idx], device=self.device))
            quat_list.append(torch.tensor(data['quat'][frame_idx], device=self.device))
            lin_vel_list.append(torch.tensor(data['lin_vel'][frame_idx], device=self.device))
            ang_vel_list.append(torch.tensor(data['ang_vel'][frame_idx], device=self.device))
            dof_list.append(torch.tensor(data['dof_pos'][frame_idx], device=self.device))
            
        keyframes["keyframe_pos"] = torch.stack(pos_list, dim=0)
        keyframes["keyframe_quat"] = torch.stack(quat_list, dim=0)
        keyframes["keyframe_lin_vel"] = torch.stack(lin_vel_list, dim=0)
        keyframes["keyframe_ang_vel"] = torch.stack(ang_vel_list, dim=0)
        keyframes["keyframe_dof_pos"] = torch.stack(dof_list, dim=0)
        
        return keyframes

def compute_residual_observations(motion_dict, base_quat, body_pos, body_quat, body_lin_vel, body_ang_vel):
    res_body_pos = quat_apply_yaw_inverse(base_quat[:, None], motion_dict["keyframe_pos"] - body_pos)
    res_body_quat = quat_mul_yaw_inverse(base_quat[:, None], quat_mul(quat_conjugate(body_quat), motion_dict["keyframe_quat"]))
    res_body_lin_vel = quat_apply_yaw_inverse(base_quat[:, None], motion_dict["keyframe_lin_vel"] - body_lin_vel)
    res_body_ang_vel = quat_apply_yaw_inverse(base_quat[:, None], motion_dict["keyframe_ang_vel"] - body_ang_vel)
    return res_body_pos, res_body_quat, res_body_lin_vel, res_body_ang_vel
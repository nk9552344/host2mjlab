from dataclasses import dataclass, field
from typing import List, Dict, Tuple

@dataclass
class StandUpEnvCfg:
    class env:
        num_envs: int = 4096
        num_observations: int = 456  # 76 single-step dimensions * 6 history steps
        num_actions: int = 29
        env_spacing: float = 3.0
        episode_length_s: float = 20.0
        
        num_one_step_observations: int = 76
        num_actor_history: int = 6
        no_orientation: bool = False

    class sim:
        dt: float = 0.005
        decimation: int = 4  # policy control rate = 50Hz (0.02s)
        gravity: List[float] = field(default_factory=lambda: [0.0, 0.0, -9.81])
        
    class control:
        control_type: str = 'P'
        # Ported PD parameters from HoST configurations
        stiffness: Dict[str, float] = field(default_factory=lambda: {
            'hip_yaw': 150.0, 'hip_roll': 150.0, 'hip_pitch': 200.0,
            'knee': 200.0, 'ankle_pitch': 20.0, 'ankle_roll': 20.0,
            'waist_yaw': 200.0, 'waist_roll': 200.0, 'waist_pitch': 200.0,
            'shoulder_pitch': 40.0, 'shoulder_roll': 40.0, 'shoulder_yaw': 40.0,
            'elbow': 40.0, 'wrist_roll': 10.0
        })
        damping: Dict[str, float] = field(default_factory=lambda: {
            'hip_yaw': 5.0, 'hip_roll': 5.0, 'hip_pitch': 5.0,
            'knee': 5.0, 'ankle_pitch': 4.0, 'ankle_roll': 4.0,
            'waist_yaw': 5.0, 'waist_roll': 5.0, 'waist_pitch': 5.0,
            'shoulder_pitch': 2.0, 'shoulder_roll': 2.0, 'shoulder_yaw': 2.0,
            'elbow': 2.0, 'wrist_roll': 1.0
        })
        action_scale: float = 0.25
        clip_actions: float = 10.0

    class init_state:
        pos: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.5])
        rot: List[float] = field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0]) # w, x, y, z format
        
        default_joint_angles: Dict[str, float] = field(default_factory=lambda: {
            'left_hip_yaw_joint': 0.0, 'left_hip_roll_joint': 0.0, 'left_hip_pitch_joint': -0.1,
            'left_knee_joint': 0.3, 'left_ankle_pitch_joint': -0.2, 'left_ankle_roll_joint': 0.0,
            'right_hip_yaw_joint': 0.0, 'right_hip_roll_joint': 0.0, 'right_hip_pitch_joint': -0.1,
            'right_knee_joint': 0.3, 'right_ankle_pitch_joint': -0.2, 'right_ankle_roll_joint': 0.0,
            'waist_yaw_joint': 0.0, 'waist_roll_joint': 0.0, 'waist_pitch_joint': 0.0,
            'left_shoulder_pitch_joint': 0.0, 'left_shoulder_roll_joint': 0.0, 'left_shoulder_yaw_joint': 0.0,
            'left_elbow_joint': 0.0, 'left_wrist_roll_joint': 0.0,
            'right_shoulder_pitch_joint': 0.0, 'right_shoulder_roll_joint': 0.0, 'right_shoulder_yaw_joint': 0.0,
            'right_elbow_joint': 0.0, 'right_wrist_roll_joint': 0.0
        })

    class rewards:
        # Multi-phase metrics defined in the HoST tasks
        target_base_height_phase1: float = 0.45
        target_base_height_phase2: float = 0.45
        target_base_height_phase3: float = 0.65
        
        reward_groups: List[str] = field(default_factory=lambda: ['task', 'regu', 'style', 'target'])
        reward_group_weights: List[float] = field(default_factory=lambda: [1.0, 0.1, 1.0, 1.0])
        
        # Core scaling metrics
        tracking_sigma: float = 0.25
        tracking_sigma_orn: float = 0.25
        
        # Penalties configuration
        penalize_contacts_on: List[str] = field(default_factory=lambda: [
            'elbow', 'shoulder', 'waist'
        ])
        contact_force_threshold: float = 1.0

    class termination:
        terminal_body_height: float = 0.25
        penalize_illegal_contacts: bool = True
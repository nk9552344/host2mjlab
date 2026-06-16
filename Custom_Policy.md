src/mjlab/tasks/standup/
│
├── __init__.py
│   # Re-exports standup_env_cfg and imports all config sub-packages
│   # so the task registry fires on mjlab import.
│
├── standup_env_cfg.py
│   # Core factory function: make_standup_env_cfg()
│   # Defines the shared ManagerBasedRlEnvCfg with:
│   #   - SceneCfg  (G1 robot entity, flat/slope/platform terrain)
│   #   - ObservationsCfg  (gravity vec, ang_vel, dof_pos, dof_vel, last_action)
│   #   - ActionsCfg  (JointPositionActionCfg, 23-DOF, action_scale=0.25)
│   #   - RewardsCfg  (all HoST reward terms imported from mdp/rewards.py)
│   #   - TerminationsCfg  (head contact, time_out)
│   #   - EventsCfg  (RSI reset — the key HoST init logic — push randomisation,
│   #                  friction rand, base mass rand)
│   #   - CurriculumCfg  (terrain difficulty scheduler)
│
├── mdp/
│   ├── __init__.py
│   │   # Exports everything from rewards.py, observations.py, events.py,
│   │   # terminations.py so callers can do:
│   │   #   from mjlab.tasks.standup import mdp
│   │
│   ├── rewards.py
│   │   # All HoST reward term functions (each returns Tensor[num_envs]):
│   │   #   height_progress()         – exp(-4 * |root_z - target_h|)
│   │   #   upright_orientation()     – projected gravity -z component
│   │   #   body_upright()            – penalise roll/pitch of pelvis
│   │   #   alive()                   – constant +1 while alive
│   │   #   action_rate()             – ||a_t - a_{t-1}||^2
│   │   #   joint_torques()           – ||tau||^2
│   │   #   joint_vel()               – ||dq||^2
│   │   #   joint_acc()               – ||ddq||^2
│   │   #   smoothness()              – ||a_t - a_{t-1}||^2  (HoST smooth reg)
│   │   #   jerk()                    – ||Δa_t - Δa_{t-1}||^2
│   │   #   undesired_contacts()      – torso / head contact penalty
│   │
│   ├── observations.py
│   │   # Custom observation terms not already in mjlab.mdp:
│   │   #   projected_gravity()       – gravity vec in body frame (IMU surrogate)
│   │   #   base_ang_vel()            – root angular velocity
│   │   #   joint_pos_rel()           – dof_pos - default_pos
│   │   #   joint_vel()               – dof vel
│   │   #   last_action()             – previous action
│   │
│   ├── events.py
│   │   # HoST-specific reset / domain-rand event functions:
│   │   #   reset_from_random_posture()  – RSI: randomly picks supine / prone /
│   │   #                                  side-left / side-right posture and sets
│   │   #                                  root qpos + qvel accordingly
│   │   #   push_robot()                 – applies random xy velocity impulse
│   │   #   randomize_joint_params()     – per-joint stiffness/damping noise
│   │   #   randomize_body_mass()        – adds ±2 kg to base link
│   │   #   randomize_friction()         – uniform geom friction in [0.5, 1.25]
│   │
│   └── terminations.py
│       # HoST termination conditions:
│       #   head_contact()   – episode ends if head geom contacts ground
│       #   time_out()       – standard episode length exceeded
│       #   bad_orientation() – (optional) kill if robot flips mid-standup
│
└── config/
    ├── __init__.py
    │   # Imports all robot sub-packages so they register on import
    │
    ├── g1/
    │   ├── __init__.py
    │   │   # Calls register_mjlab_task() for each variant:
    │   │   #   "Mjlab-StandUp-Ground-Unitree-G1"
    │   │   #   "Mjlab-StandUp-Ground-Prone-Unitree-G1"
    │   │   #   "Mjlab-StandUp-Slope-Unitree-G1"
    │   │   #   "Mjlab-StandUp-Platform-Unitree-G1"
    │   │   #   "Mjlab-StandUp-Wall-Unitree-G1"
    │   │
    │   └── env_cfgs.py
    │       # Five factory functions (one per terrain scenario), each calling
    │       # make_standup_env_cfg() and overriding terrain-specific params,
    │       # plus a *_play variant (noise off, 1 env, infinite episode):
    │       #   g1_standup_ground_env_cfg()        – flat ground, all postures
    │       #   g1_standup_ground_prone_env_cfg()  – prone-only RSI
    │       #   g1_standup_slope_env_cfg()         – inclined terrain
    │       #   g1_standup_platform_env_cfg()      – raised platform
    │       #   g1_standup_wall_env_cfg()          – wall-lean start
    │       #   *_play_env_cfg() variants for each (noise off, 1 env)
    │
    └── h1/
        ├── __init__.py
        │   # Calls register_mjlab_task() for:
        │   #   "Mjlab-StandUp-Ground-Unitree-H1"
        │
        └── env_cfgs.py
            # h1_standup_ground_env_cfg() – same pattern, H1 robot entity
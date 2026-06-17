from dataclasses import dataclass, field
from typing import List

@dataclass
class G1StandUpPPOCfg:
    seed: int = 1
    runner_class_name: str = 'OnPolicyRunner'
    
    class policy:
        init_noise_std: float = 0.8
        actor_hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])
        critic_hidden_dims: List[int] = field(default_factory=lambda: [512, 256])
        activation: str = 'elu'
        
    class algorithm:
        value_loss_coef: float = 1.0
        use_clipped_value_loss: bool = True
        clip_param: float = 0.2
        entropy_coef: float = 0.01
        num_learning_epochs: int = 5
        num_mini_batches: int = 4
        learning_rate: float = 1e-5
        schedule: str = 'adaptive'
        gamma: float = 0.99
        lam: float = 0.95
        desired_kl: float = 0.01
        max_grad_norm: float = 1.0

    class runner:
        policy_class_name: str = 'ActorCritic'
        algorithm_class_name: str = 'PPO'
        num_steps_per_env: int = 24
        max_iterations: int = 1500
        save_interval: int = 100
        experiment_name: str = 'g1_standup'
        run_name: str = ''
        resume: bool = False
        load_run: str = '-1'
        checkpoint: int = -1
        resume_path: str = ''
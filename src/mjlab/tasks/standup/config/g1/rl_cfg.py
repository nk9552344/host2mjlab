"""RL configuration for Unitree G1 standup task.

Forked from the velocity locomotion RL cfg. This file is almost entirely
training-infrastructure hyperparameters with no direct coupling to task
semantics (network architecture, PPO mechanics), so most values are carried
over unchanged. Two changes:

  - experiment_name: "g1_velocity" -> "g1_standup", since it was simply
    stale, not a deliberate choice.
  - entropy_coef: 0.01 -> 0.03. Locomotion mostly explores variations
    around a single stable walking gait; standup needs to discover a
    qualitatively different recovery strategy from a wide range of fallen
    starting states (face-down, face-up, on a side, mid-tumble after a
    push), so more exploration pressure early in training is a reasonable
    starting point. This is a starting guess, not a tuned value -- treat it
    as the first thing to revisit if training stalls into a single rigid
    "flailing" policy or, conversely, fails to converge at all.

Everything else (hidden_dims, activation, obs_normalization,
distribution_cfg, clip_param, gamma, lam, learning_rate, schedule,
desired_kl, max_grad_norm, num_steps_per_env, max_iterations, save_interval)
left unchanged -- no strong signal these need to differ for standup, and
guessing new values without tuning data would be worse than keeping known-
reasonable defaults. Revisit num_steps_per_env/max_iterations once you have
a sense of how standup's convergence behavior compares to locomotion's.
"""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def unitree_g1_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for Unitree G1 standup task."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      # Smaller network to match the proven simple stay_stand recipe.
      # 256-128-64 has enough capacity for a balance policy and trains
      # noticeably faster than the 512-256-128 inherited from the
      # locomotion fork. The bigger network was over-parameterised for
      # this task and slowed gradient updates.
      hidden_dims=(256, 128, 64),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        # FREEZE THE POLICY STD. Across every previous run, Policy/mean_std
        # grew monotonically (0.1 -> 0.18 -> 0.28...) regardless of
        # entropy_coef (tried 0.03, 0.005, 0.001, 0.0). The mechanism is
        # PPO's surrogate gradient on log_std: with noisy bounded rewards
        # like standup_progress, outlier actions occasionally hit positive
        # advantage, which pushes log_std up. There is no counter-force
        # (entropy_coef=0 doesn't shrink std, it just removes the bias to
        # grow it). The growing std then physically prevents the robot
        # from holding the standing pose, so the exp-form rewards (pose,
        # upright, hold_still) collapse to ~0, leaving only the noisy
        # standup_progress signal -- a self-reinforcing failure mode.
        #
        # learn_std=False fixes std at init_std and removes log_std from
        # the optimizer entirely, breaking the feedback loop. The policy
        # must improve by shaping the *mean* action, which is exactly
        # what we want for a "hold default pose" task.
        #
        # init_std=0.2 (was 0.4 in session 5 when the task included
        # active pushes and required dynamic balance discovery). With
        # pushes disabled and the task reduced to "just stand still",
        # 0.4 is excessive exploration -- per-step action perturbation
        # of ~0.4×0.25×0.35rad ≈ 0.035 rad/joint is enough to learn
        # tiny corrective motions but small enough not to constantly
        # destabilise the robot before the policy converges. The
        # earlier "we need stepping discovery" rationale no longer
        # applies for the minimal balance task.
        "init_std": 0.2,
        "std_type": "scalar",
        "learn_std": False,
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(256, 128, 64),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=0.5,
      use_clipped_value_loss=True,
      clip_param=0.2,
      # entropy_coef=0.001 still pushed std upward against a weak reward
      # signal (entropy climbed from -4 to +2 across the last run). With
      # init_std=0.1 the policy already has enough exploration; setting
      # this to 0 removes any bias toward making the policy noisier.
      entropy_coef=0.0,
      # Adaptive KL schedule collapsed the LR to its hardcoded floor
      # (~1e-5) within 1 iteration on every previous run and never
      # recovered -- visible as a flat Loss/learning_rate line at 1.1e-5.
      # Switching to a fixed schedule decouples training from the noisy
      # early-training KL signal. 3e-4 is a conservative LR that keeps
      # updates small enough to be stable without the adaptive-KL
      # throttling pathology.
      learning_rate=3.0e-4,
      schedule="fixed",
      num_learning_epochs=5,
      num_mini_batches=4,
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,  # Unused under schedule="fixed" but kept for clarity.
      max_grad_norm=1.0,
    ),
    experiment_name="g1_staystand",
    save_interval=50,
    # ROOT-CAUSE FIX. Previously this was 24 (= 0.48 s of rollout). With
    # episodes terminating at ~74 control steps (~1.5 s), every rollout
    # ended INSIDE a still-standing episode -- the fell_over termination
    # never landed in the same buffer as the actions that led to it.
    # GAE therefore computed advantages on episode fragments, the critic
    # never saw the causal chain action -> fall, and the policy gradient
    # was effectively noise. Train/mean_reward declined monotonically
    # while Train/mean_episode_length sat fixed at ~74 steps.
    #
    # This exact pathology and its fix are documented in the simple
    # stay_stand task's agent_context.txt (section 5, Diagnosis 3). The
    # proven value of 128 (= 2.56 s) covers a full episode end-to-end so
    # the critic can attribute terminations to the actions that caused
    # them.
    num_steps_per_env=128,
    max_iterations=30_000,
  )
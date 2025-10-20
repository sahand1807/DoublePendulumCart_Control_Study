"""
Train PPO for swing-up task (starting from hanging position).

The swing-up problem is fundamentally different from stabilization:
- Initial state: θ₁=0, θ₂=0 (hanging down) instead of θ≈π (near upright)
- Task: Swing up AND stabilize in one continuous motion
- Reward: Unified quadratic cost that works for both swing-up and stabilization
- Difficulty: Much harder than stabilization alone

This script trains a policy from scratch using the swing-up environment.

Usage:
    # Train with default settings (1M timesteps)
    python experiments/train_ppo_swingup.py

    # Train with custom timesteps
    python experiments/train_ppo_swingup.py --timesteps 2000000

    # Adjust episode length
    python experiments/train_ppo_swingup.py --max-episode-steps 10000

Reference:
    Based on IROS 2024 AI Olympics winner approach (AR-EAPO)
    "Average-Reward Maximum Entropy RL for Underactuated Double Pendulum Tasks"
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, EvalCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

from env.double_pendulum_cart_env import DoublePendulumCartEnv
from env.angle_wrapper import AngleObservationWrapper
from env.swing_up_wrapper import SwingUpInitializationWrapper, SwingUpRewardWrapper


def create_swing_up_env(n_envs=1, seed=42, monitor=True, max_episode_steps=10000):
    """
    Create wrapped environment for swing-up training.

    The environment stack:
    1. DoublePendulumCartEnv (base physics)
    2. SwingUpInitializationWrapper (start at θ=0,0)
    3. SwingUpRewardWrapper (unified quadratic reward)
    4. AngleObservationWrapper (sin/cos encoding)
    5. Monitor (optional, for logging)

    Args:
        n_envs: Number of parallel environments
        seed: Random seed
        monitor: Whether to wrap with Monitor for logging
        max_episode_steps: Maximum steps per episode (swing-up needs longer episodes)

    Returns:
        Vectorized environment ready for PPO training
    """

    def make_env(rank: int):
        """Factory function to create single environment."""

        def _init():
            # Create base environment with longer episodes for swing-up
            # IMPORTANT: terminate_on_fall=False to allow swinging from hanging position
            # High force (100N) needed for swing-up from hanging
            env = DoublePendulumCartEnv(terminate_on_fall=False, max_force=100.0)
            env.max_episode_steps = max_episode_steps

            # Apply swing-up wrappers
            env = SwingUpInitializationWrapper(env, perturbation_range=0.1)
            env = SwingUpRewardWrapper(
                env,
                angle_weight=50.0,           # High priority on reaching upright
                velocity_weight_theta1=4.0,  # Moderate damping
                velocity_weight_theta2=2.0,
                position_weight=1.0,         # Keep cart near center
                cart_velocity_weight=0.5,
                control_weight=1.0,          # Penalize excessive force
                scale=0.001                  # Overall reward scaling
            )

            # Apply angle encoding (sin/cos)
            env = AngleObservationWrapper(env)

            if monitor:
                env = Monitor(env)

            env.reset(seed=seed + rank)
            return env

        return _init

    # Create vectorized environment
    if n_envs == 1:
        env = DummyVecEnv([make_env(0)])
    else:
        env = SubprocVecEnv([make_env(i) for i in range(n_envs)])

    return env


class SwingUpSuccessCallback(BaseCallback):
    """
    Callback to track swing-up success rate during training.

    Success criterion: Episode reaches near-upright position and stays there
    - Final angle error < 10° (0.175 rad) for both pendulums
    - Episode length > 90% of max (indicating stability, not early termination)
    """

    def __init__(self, max_episode_steps=10000, verbose=0):
        super().__init__(verbose)
        self.max_episode_steps = max_episode_steps
        self.episode_angle_errors = []
        self.episode_lengths = []
        self.success_threshold = np.deg2rad(10)  # 10 degrees

    def _on_step(self) -> bool:
        # Check if episode ended
        if self.locals.get("dones") is not None:
            dones = self.locals["dones"]
            infos = self.locals["infos"]

            for i, done in enumerate(dones):
                if done:
                    # Get episode info
                    if "episode" in infos[i]:
                        episode_length = infos[i]["episode"]["l"]
                        self.episode_lengths.append(episode_length)

                    # Get final angle error
                    if "upright_error" in infos[i]:
                        error = infos[i]["upright_error"]
                        self.episode_angle_errors.append(error)

        return True

    def _on_rollout_end(self) -> None:
        if len(self.episode_angle_errors) > 0 and len(self.episode_lengths) > 0:
            # Success = reached near-upright AND stayed stable
            min_length = int(0.9 * self.max_episode_steps)
            successes = [
                (err < self.success_threshold and length > min_length)
                for err, length in zip(self.episode_angle_errors, self.episode_lengths)
            ]
            success_rate = np.mean(successes) * 100 if successes else 0.0

            self.logger.record("rollout/swing_up_success_rate", success_rate)
            self.logger.record("rollout/mean_angle_error", np.mean(self.episode_angle_errors))
            self.logger.record("rollout/mean_episode_length", np.mean(self.episode_lengths))

            # Clear for next rollout
            self.episode_angle_errors = []
            self.episode_lengths = []


def train_swing_up(total_timesteps=1_000_000, max_episode_steps=10000):
    """
    Train PPO for swing-up task.

    Args:
        total_timesteps: Total training timesteps (default: 1M)
        max_episode_steps: Maximum steps per episode (default: 10000 = 50 seconds)
    """
    print("\n" + "="*70)
    print("PPO SWING-UP TRAINING: θ₁=0°, θ₂=0° → θ₁=180°, θ₂=180°")
    print("="*70)
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Max episode steps: {max_episode_steps:,} ({max_episode_steps*0.005:.1f} seconds)")
    print(f"Parallel environments: 4")
    print(f"Evaluation frequency: 10,000 steps")
    print(f"Task: Swing up from hanging position AND stabilize")
    print("="*70 + "\n")

    # Setup directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    results_dir = os.path.join(project_root, 'results/ppo_swingup')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'best_model'), exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'eval'), exist_ok=True)

    # Create environments
    print("Creating training environment (4 parallel)...")
    train_env = create_swing_up_env(
        n_envs=4,
        seed=42,
        monitor=True,
        max_episode_steps=max_episode_steps
    )

    print("Creating evaluation environment...")
    eval_env = create_swing_up_env(
        n_envs=1,
        seed=999,
        monitor=True,
        max_episode_steps=max_episode_steps
    )

    # Create PPO model with hyperparameters tuned for swing-up
    # Swing-up is harder than stabilization, so we use:
    # - Higher entropy coefficient for exploration
    # - Larger network for complex dynamics
    # - Conservative learning rate for stable learning
    print("\n🆕 Creating new PPO model for swing-up...")
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,  # Exploration is critical for discovering swing-up
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs={"net_arch": [256, 256]},  # Larger network for complex task
        verbose=1,
        tensorboard_log=results_dir,
        seed=42,
    )
    print(f"   📉 Learning rate: 3e-4")
    print(f"   🎲 Entropy coefficient: 0.01 (encourages exploration)")
    print(f"   📦 Batch size: 64")
    print(f"   🧠 Network: [256, 256] (larger for swing-up complexity)")
    print("   ✅ Model created\n")

    # Setup callbacks
    success_callback = SwingUpSuccessCallback(
        max_episode_steps=max_episode_steps,
        verbose=1
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(results_dir, 'best_model'),
        log_path=os.path.join(results_dir, 'eval'),
        eval_freq=10_000,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
        verbose=1,
    )
    callbacks = CallbackList([success_callback, eval_callback])

    # Train
    print("🚀 Starting training...\n")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True,
    )

    # Save final model
    final_model_path = os.path.join(results_dir, 'ppo_swingup_final')
    model.save(final_model_path)
    print(f"\n✅ Training complete!")
    print(f"   Final model saved to: {final_model_path}.zip")

    # Cleanup
    train_env.close()
    eval_env.close()

    return final_model_path + '.zip'


def main():
    parser = argparse.ArgumentParser(
        description="Train PPO for swing-up task (θ=0 → θ=π)"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=1_000_000,
        help="Total training timesteps (default: 1,000,000)",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=10_000,
        help="Maximum steps per episode (default: 10,000 = 50 seconds)",
    )

    args = parser.parse_args()

    # Train
    model_path = train_swing_up(
        total_timesteps=args.timesteps,
        max_episode_steps=args.max_episode_steps,
    )

    print("\n" + "="*70)
    print("🎉 SWING-UP TRAINING COMPLETE!")
    print("="*70)
    print(f"\nTrained model: {model_path}")
    print(f"Task: Swing up from hanging (θ=0°) to inverted (θ=180°)")
    print("\nNext steps:")
    print("  1. Evaluate: python experiments/evaluate_ppo_swingup.py")
    print("  2. Visualize: python experiments/render_ppo_swingup.py")
    print("  3. Compare with stabilization controllers (Level 1/2/3)")
    print()


if __name__ == "__main__":
    main()

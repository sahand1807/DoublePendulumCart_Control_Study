"""
Train PPO Level 2 (±6° perturbations) with transfer learning from Level 1.

Usage:
    # Train with transfer learning from Level 1 best model (recommended)
    python experiments/train_ppo_level2.py --transfer-from results/ppo_level1/best_model/best_model.zip

    # Train from scratch (not recommended)
    python experiments/train_ppo_level2.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, EvalCallback

from controllers.ppo_controller import create_ppo_env, SuccessRateCallback


def linear_schedule(initial_value: float, final_value: float):
    """
    Linear learning rate schedule with warmup and decay.

    Args:
        initial_value: Initial learning rate
        final_value: Final learning rate

    Returns:
        Schedule function that takes progress (0-1) and returns LR
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will go from 1 (beginning) to 0 (end).
        We want: warmup for first 10%, then linear decay.
        """
        progress = 1.0 - progress_remaining  # Convert to 0 (start) -> 1 (end)

        warmup_fraction = 0.1
        if progress < warmup_fraction:
            # Linear warmup: 0 -> initial_value
            return (progress / warmup_fraction) * initial_value
        else:
            # Linear decay: initial_value -> final_value
            decay_progress = (progress - warmup_fraction) / (1.0 - warmup_fraction)
            return initial_value + (final_value - initial_value) * decay_progress

    return func


def train_level2(transfer_model_path=None, total_timesteps=200_000):
    """
    Train PPO Level 2 controller (±6°).

    Args:
        transfer_model_path: Path to Level 1 model for transfer learning (required)
        total_timesteps: Total training timesteps (default: 200K)
    """
    print("\n" + "="*70)
    print("PPO LEVEL 2 TRAINING: ±6° Perturbations (2x Level 1)")
    print("="*70)
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Parallel environments: 4")
    print(f"Evaluation frequency: 10,000 steps")
    if transfer_model_path:
        print(f"Transfer learning from: {transfer_model_path}")
    else:
        print("⚠️  WARNING: Training from scratch not recommended!")
        print("   Use --transfer-from for better results")
    print("="*70 + "\n")

    # Setup directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    results_dir = os.path.join(project_root, 'results/ppo_level2')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'best_model'), exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'eval'), exist_ok=True)

    # Create environments
    print("Creating training environment (4 parallel)...")
    train_env = create_ppo_env(curriculum_level=2, n_envs=4, seed=42, monitor=True)

    print("Creating evaluation environment...")
    eval_env = create_ppo_env(curriculum_level=2, n_envs=1, seed=999, monitor=True)

    # Transfer learning with conservative hyperparameters:
    # 1. Lower learning rate to fine-tune without destroying Level 1 knowledge
    # 2. Keep entropy coefficient for exploration in slightly harder environment
    # 3. Use same batch size as Level 1 for consistency

    if transfer_model_path and os.path.exists(transfer_model_path):
        print(f"\n🔄 Loading Level 1 model for transfer learning...")
        print(f"   Model: {transfer_model_path}")
        model = PPO.load(transfer_model_path, env=train_env)

        # Increased learning rate for better adaptation to ±6° perturbations
        # Balanced to preserve Level 1 knowledge while allowing exploration
        initial_lr = 1.5e-4  # Increased from 5e-5 for faster adaptation
        final_lr = 3e-5      # Increased from 1e-5 for continued learning
        model.learning_rate = linear_schedule(initial_lr, final_lr)

        # Keep entropy coefficient for exploration
        model.ent_coef = 0.01

        # Keep batch size consistent with Level 1
        model.batch_size = 64

        print(f"   📉 LR schedule: {initial_lr:.0e} → {final_lr:.0e} (balanced)")
        print(f"   🎲 Entropy coefficient: {model.ent_coef}")
        print(f"   📦 Batch size: {model.batch_size}")
        print(f"   🎯 Strategy: Gradual adaptation from ±3° to ±6°")
        print("   ✅ Model loaded successfully\n")
    else:
        print("\n🆕 Creating new PPO model...")
        # For training from scratch, use standard LR with schedule
        initial_lr = 3e-4
        final_lr = 3e-5
        model = PPO(
            policy="MlpPolicy",
            env=train_env,
            learning_rate=linear_schedule(initial_lr, final_lr),
            n_steps=2048,
            batch_size=128,  # Increased for stability
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,  # Start with exploration
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs={"net_arch": [64, 64]},
            verbose=1,
            tensorboard_log=results_dir,
            seed=42,  # For reproducibility
        )
        print(f"   📉 LR schedule: {initial_lr:.0e} → {final_lr:.0e} (10% warmup)")
        print(f"   🎲 Entropy coefficient: {model.ent_coef}")
        print(f"   📦 Batch size: {model.batch_size}")
        print("   ✅ Model created\n")

    # Setup callbacks
    success_callback = SuccessRateCallback(verbose=1)
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
    final_model_path = os.path.join(results_dir, 'ppo_level2_final')
    model.save(final_model_path)
    print(f"\n✅ Training complete!")
    print(f"   Final model saved to: {final_model_path}.zip")

    # Cleanup
    train_env.close()
    eval_env.close()

    return final_model_path + '.zip'


def main():
    parser = argparse.ArgumentParser(
        description="Train PPO Level 2 (±10° perturbations)"
    )
    parser.add_argument(
        "--transfer-from",
        type=str,
        default=None,
        help="Path to Level 1 model for transfer learning (e.g., results/ppo_level1/best_model/best_model.zip)",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=200_000,
        help="Total training timesteps (default: 200,000)",
    )

    args = parser.parse_args()

    # Train
    model_path = train_level2(
        transfer_model_path=args.transfer_from,
        total_timesteps=args.timesteps,
    )

    print("\n" + "="*70)
    print("🎉 LEVEL 2 TRAINING COMPLETE!")
    print("="*70)
    print(f"\nTrained model: {model_path}")
    print(f"Curriculum progression: Level 1 (±3°) → Level 2 (±6°)")
    print("\nNext steps:")
    print("  1. Evaluate: python experiments/evaluate_ppo_level2.py")
    print("  2. Visualize: python experiments/render_ppo_level2.py")
    print("  3. Train Level 3 (±10°): Use this model as baseline")
    print()


if __name__ == "__main__":
    main()

"""
Train PPO Level 2 (±10° perturbations) with optional transfer learning from Level 1.

Usage:
    # Train from scratch
    python experiments/train_ppo_level2.py

    # Train with transfer learning from Level 1
    python experiments/train_ppo_level2.py --transfer-from results/ppo_level1/ppo_level1_final.zip
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, EvalCallback

from controllers.ppo_controller import create_ppo_env, SuccessRateCallback


def train_level2(transfer_model_path=None, total_timesteps=500_000):
    """
    Train PPO Level 2 controller.

    Args:
        transfer_model_path: Path to Level 1 model for transfer learning (optional)
        total_timesteps: Total training timesteps (default: 500K)
    """
    print("\n" + "="*70)
    print("PPO LEVEL 2 TRAINING: ±10° Perturbations")
    print("="*70)
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Parallel environments: 4")
    print(f"Evaluation frequency: 10,000 steps")
    if transfer_model_path:
        print(f"Transfer learning from: {transfer_model_path}")
    else:
        print("Training from scratch (no transfer learning)")
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

    # Create or load model
    if transfer_model_path and os.path.exists(transfer_model_path):
        print(f"\n🔄 Loading Level 1 model for transfer learning...")
        print(f"   Model: {transfer_model_path}")
        model = PPO.load(transfer_model_path, env=train_env)
        print("   ✅ Model loaded successfully\n")
    else:
        print("\n🆕 Creating new PPO model...")
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
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs={"net_arch": [64, 64]},
            verbose=1,
            tensorboard_log=results_dir,
        )
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
        help="Path to Level 1 model for transfer learning (e.g., results/ppo_level1/ppo_level1_final.zip)",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=500_000,
        help="Total training timesteps (default: 500,000)",
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
    print("\nNext steps:")
    print("  1. Evaluate: python experiments/evaluate_ppo_level2.py")
    print("  2. Visualize: python experiments/render_ppo_level2.py")
    print("  3. Train Level 3: python experiments/train_ppo_level3.py --transfer-from", model_path)
    print()


if __name__ == "__main__":
    main()

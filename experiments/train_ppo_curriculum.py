"""
PPO Curriculum Training for Double Pendulum Cart Stabilization

This script implements progressive curriculum learning:
- Level 1 (±3°): Easy - Train from scratch
- Level 2 (±10°): Medium - Continue from Level 1 checkpoint
- Level 3 (±30°): Hard - Continue from Level 2 checkpoint

The curriculum approach helps PPO learn incrementally, building on
previously acquired skills to handle progressively larger perturbations.
"""
import os
import sys
import argparse
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from controllers.ppo_controller import (
    PPOController,
    create_ppo_env,
    SuccessRateCallback,
)
from stable_baselines3.common.callbacks import CallbackList, EvalCallback


# Curriculum configuration
CURRICULUM_CONFIG = {
    1: {
        "name": "Level_1_Easy_3deg",
        "angle_range_deg": 3,
        "timesteps": 500_000,  # 500K timesteps for proper convergence
        "n_envs": 4,  # 4 parallel environments
        "eval_freq": 10_000,  # Eval every 10K steps
        "description": "±3° perturbations - Easy stabilization",
    },
    2: {
        "name": "Level_2_Medium_10deg",
        "angle_range_deg": 10,
        "timesteps": 500_000,  # 500K for medium difficulty
        "n_envs": 4,
        "eval_freq": 10_000,
        "description": "±10° perturbations - Medium difficulty",
    },
    3: {
        "name": "Level_3_Hard_30deg",
        "angle_range_deg": 30,
        "timesteps": 500_000,  # 500K for hardest level
        "n_envs": 4,
        "eval_freq": 10_000,
        "description": "±30° perturbations - Hard stabilization",
    },
}


def train_curriculum_level(
    level: int,
    results_dir: str,
    previous_model_path: str = None,
    verbose: bool = True,
):
    """
    Train PPO agent at a specific curriculum level.

    Args:
        level: Curriculum level (1, 2, or 3)
        results_dir: Directory to save results and checkpoints
        previous_model_path: Path to model from previous level (for transfer learning)
        verbose: Print progress information

    Returns:
        Path to trained model
    """
    config = CURRICULUM_CONFIG[level]

    if verbose:
        print("\n" + "=" * 80)
        print(f"CURRICULUM LEVEL {level}: {config['name']}")
        print("=" * 80)
        print(f"Description: {config['description']}")
        print(f"Training timesteps: {config['timesteps']:,}")
        print(f"Parallel environments: {config['n_envs']}")
        print(f"Evaluation frequency: {config['eval_freq']:,}")
        if previous_model_path:
            print(f"Transfer learning from: {previous_model_path}")
        print("=" * 80 + "\n")

    # Create directories
    level_dir = os.path.join(results_dir, config["name"])
    os.makedirs(level_dir, exist_ok=True)
    os.makedirs(os.path.join(level_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(level_dir, "eval"), exist_ok=True)

    # Create training environment (vectorized for speed)
    print(f"Creating training environment with {config['n_envs']} parallel instances...")
    train_env = create_ppo_env(
        curriculum_level=level,
        n_envs=config["n_envs"],
        seed=42,
        monitor=True,
    )

    # Create evaluation environment (single env for deterministic eval)
    print(f"Creating evaluation environment...")
    eval_env = create_ppo_env(
        curriculum_level=level,
        n_envs=1,
        seed=999,  # Different seed for evaluation
        monitor=True,
    )

    # Initialize or load controller
    if previous_model_path and os.path.exists(previous_model_path):
        print(f"\n🔄 Loading previous model for transfer learning...")
        print(f"   Model: {previous_model_path}")
        controller = PPOController(
            name=config["name"],
            model_path=previous_model_path,
        )
        # Update the environment for the loaded policy
        controller.policy.set_env(train_env)
        print("   ✅ Model loaded successfully\n")
    else:
        print(f"\n🆕 Creating new PPO controller...")
        controller = PPOController(name=config["name"])
        print("   ✅ Controller created\n")

    # Setup callbacks
    success_callback = SuccessRateCallback(verbose=1)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(level_dir, "best_model"),
        log_path=os.path.join(level_dir, "eval"),
        eval_freq=config["eval_freq"],
        n_eval_episodes=10,
        deterministic=True,
        render=False,
        verbose=1,
    )

    callbacks = CallbackList([success_callback, eval_callback])

    # Train the agent
    print(f"🚀 Starting training...\n")
    controller.train(
        env=train_env,
        total_timesteps=config["timesteps"],
        eval_freq=config["eval_freq"],
        save_path=level_dir,
        callback=callbacks,
    )

    # Save final model
    final_model_path = os.path.join(level_dir, f"{config['name']}_final.zip")
    controller.save(final_model_path)

    # Clean up
    train_env.close()
    eval_env.close()

    print(f"\n✅ Level {level} training complete!")
    print(f"   Final model saved to: {final_model_path}\n")

    return final_model_path


def evaluate_level(
    level: int,
    model_path: str,
    n_episodes: int = 50,
    verbose: bool = True,
):
    """
    Evaluate trained model at a specific curriculum level.

    Args:
        level: Curriculum level (1, 2, or 3)
        model_path: Path to trained model
        n_episodes: Number of evaluation episodes
        verbose: Print results

    Returns:
        Dictionary with evaluation metrics
    """
    config = CURRICULUM_CONFIG[level]

    if verbose:
        print(f"\n📊 Evaluating {config['name']}...")
        print(f"   Model: {model_path}")
        print(f"   Episodes: {n_episodes}\n")

    # Load controller
    controller = PPOController(name=config["name"], model_path=model_path)

    # Create evaluation environment
    env = create_ppo_env(curriculum_level=level, n_envs=1, seed=999, monitor=False)

    # Run evaluation episodes
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    success_threshold = np.deg2rad(3)  # 3 degrees final error

    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        ep_reward = 0
        ep_length = 0

        while not done:
            # Get action from wrapped observation (8D)
            # But controller expects original 6D state, so we need to reconstruct angles
            obs_unwrapped = obs[0]  # Remove vectorization dimension
            x = obs_unwrapped[0]
            sin_theta1, cos_theta1 = obs_unwrapped[1], obs_unwrapped[2]
            sin_theta2, cos_theta2 = obs_unwrapped[3], obs_unwrapped[4]
            dx, dtheta1, dtheta2 = obs_unwrapped[5], obs_unwrapped[6], obs_unwrapped[7]

            # Reconstruct angles
            theta1 = np.arctan2(sin_theta1, cos_theta1)
            theta2 = np.arctan2(sin_theta2, cos_theta2)

            # Create original state for controller
            state = np.array([x, theta1, theta2, dx, dtheta1, dtheta2])

            # Get control action
            action = controller.compute_control(state)

            # Step environment
            obs, reward, done, info = env.step(action)
            ep_reward += reward[0]
            ep_length += 1

            if done[0]:
                break

        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_length)

        # Check success (final angle error < 3°)
        final_x = obs[0][0]
        final_theta1 = np.arctan2(obs[0][1], obs[0][2])
        final_theta2 = np.arctan2(obs[0][3], obs[0][4])

        theta1_error = abs(final_theta1 - np.pi)
        theta2_error = abs(final_theta2 - np.pi)

        # Wrap errors
        theta1_error = min(theta1_error, 2*np.pi - theta1_error)
        theta2_error = min(theta2_error, 2*np.pi - theta2_error)

        if theta1_error < success_threshold and theta2_error < success_threshold:
            success_count += 1

    env.close()

    # Compute statistics
    results = {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "success_rate": success_count / n_episodes * 100,
    }

    if verbose:
        print(f"\n   Results:")
        print(f"   - Mean reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"   - Mean episode length: {results['mean_length']:.1f}")
        print(f"   - Success rate: {results['success_rate']:.1f}%")
        print(f"   - Successes: {success_count}/{n_episodes}\n")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Train PPO agent with curriculum learning"
    )
    parser.add_argument(
        "--start-level",
        type=int,
        default=1,
        choices=[1, 2, 3],
        help="Starting curriculum level (1, 2, or 3)",
    )
    parser.add_argument(
        "--end-level",
        type=int,
        default=3,
        choices=[1, 2, 3],
        help="Ending curriculum level (1, 2, or 3)",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/ppo_curriculum",
        help="Directory to save training results",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate trained models after training",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=50,
        help="Number of episodes for evaluation",
    )

    args = parser.parse_args()

    # Create results directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(args.results_dir, timestamp)
    os.makedirs(results_dir, exist_ok=True)

    print("\n" + "🎓" * 40)
    print("PPO CURRICULUM LEARNING - DOUBLE PENDULUM CART")
    print("🎓" * 40)
    print(f"\nResults directory: {results_dir}")
    print(f"Training levels: {args.start_level} → {args.end_level}")
    print("\n" + "=" * 80)

    # Train curriculum levels
    previous_model = None
    trained_models = {}

    for level in range(args.start_level, args.end_level + 1):
        model_path = train_curriculum_level(
            level=level,
            results_dir=results_dir,
            previous_model_path=previous_model,
            verbose=True,
        )
        trained_models[level] = model_path
        previous_model = model_path  # Use for next level

    print("\n" + "=" * 80)
    print("🎉 CURRICULUM TRAINING COMPLETE!")
    print("=" * 80)

    # Evaluate if requested
    if args.evaluate:
        print("\n" + "📊" * 40)
        print("EVALUATION")
        print("📊" * 40)

        for level in range(args.start_level, args.end_level + 1):
            results = evaluate_level(
                level=level,
                model_path=trained_models[level],
                n_episodes=args.eval_episodes,
                verbose=True,
            )

    print("\n" + "=" * 80)
    print("✅ ALL DONE!")
    print("=" * 80)
    print(f"\nTrained models saved in: {results_dir}")
    for level, path in trained_models.items():
        print(f"  Level {level}: {path}")
    print("\n")


if __name__ == "__main__":
    main()

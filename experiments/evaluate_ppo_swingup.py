"""
Evaluate trained PPO swing-up model and generate plots.

Tests the model starting from hanging position (θ=0°) and measures:
- Success rate (reaching and maintaining upright position)
- Average swing-up time
- Final stabilization quality

Success criteria for swing-up task:
1. Reaches near-upright (both angles within 10° of vertical)
2. Maintains stability (episode length ≥ 90% of max steps)
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from env.double_pendulum_cart_env import DoublePendulumCartEnv
from env.angle_wrapper import AngleObservationWrapper
from env.swing_up_wrapper import SwingUpInitializationWrapper, SwingUpRewardWrapper


def create_swingup_env(max_episode_steps=10000):
    """Create swing-up environment."""
    # IMPORTANT: terminate_on_fall=False to allow swinging from hanging position
    # High force (100N) needed for swing-up
    env = DoublePendulumCartEnv(terminate_on_fall=False, max_force=100.0)
    env.max_episode_steps = max_episode_steps
    env = SwingUpInitializationWrapper(env, perturbation_range=0.1)
    env = SwingUpRewardWrapper(env)
    env = AngleObservationWrapper(env)
    return env


def evaluate_model(model_path, n_episodes=100, max_episode_steps=10000):
    """
    Evaluate swing-up model and collect metrics.

    Args:
        model_path: Path to trained model
        n_episodes: Number of test episodes
        max_episode_steps: Maximum steps per episode

    Returns:
        Dictionary with evaluation statistics
    """
    print(f"\n{'='*70}")
    print(f"SWING-UP EVALUATION")
    print(f"{'='*70}")
    print(f"Loading model: {model_path}")
    print(f"Testing on {n_episodes} episodes...")
    print(f"Max episode steps: {max_episode_steps:,} ({max_episode_steps*0.005:.1f} seconds)")
    print(f"{'='*70}\n")

    model = PPO.load(model_path)
    env = create_swingup_env(max_episode_steps=max_episode_steps)

    # Storage for metrics
    all_lengths = []
    all_rewards = []
    swing_up_times = []  # Time to first reach upright
    successful_episodes = 0

    # Success thresholds
    angle_threshold = np.deg2rad(10)  # Within 10° of upright
    min_stable_length = int(0.9 * max_episode_steps)  # Stayed stable for 90% of episode

    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_length = 0
        episode_reward = 0.0
        swing_up_time = None

        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_length += 1
            episode_reward += reward

            # Check if reached upright for the first time
            if swing_up_time is None and 'upright_error' in info:
                if info['upright_error'] < angle_threshold:
                    swing_up_time = episode_length * 0.005  # Convert to seconds

        all_lengths.append(episode_length)
        all_rewards.append(episode_reward)

        if swing_up_time is not None:
            swing_up_times.append(swing_up_time)

        # Success: reached upright AND stayed stable
        final_error = info.get('upright_error', np.inf)
        if final_error < angle_threshold and episode_length >= min_stable_length:
            successful_episodes += 1

        # Print progress every 10 episodes
        if (episode + 1) % 10 == 0:
            print(f"Progress: {episode+1}/{n_episodes} episodes completed")

    env.close()

    # Compute statistics
    success_rate = 100 * successful_episodes / n_episodes
    mean_swing_up_time = np.mean(swing_up_times) if swing_up_times else None
    swing_up_success_rate = 100 * len(swing_up_times) / n_episodes

    # Print summary statistics
    print(f"\n{'='*70}")
    print(f"SWING-UP EVALUATION SUMMARY (n={n_episodes} episodes)")
    print(f"{'='*70}")
    print(f"Full Success Rate:    {success_rate:6.1f}% ({successful_episodes}/{n_episodes})")
    print(f"  (Reached upright AND maintained stability)")
    print()
    print(f"Swing-Up Rate:        {swing_up_success_rate:6.1f}% ({len(swing_up_times)}/{n_episodes})")
    print(f"  (Reached within {np.rad2deg(angle_threshold):.0f}° of upright)")
    if swing_up_times:
        print(f"Mean Swing-Up Time:   {mean_swing_up_time:7.2f} seconds")
        print(f"Min Swing-Up Time:    {np.min(swing_up_times):7.2f} seconds")
        print(f"Max Swing-Up Time:    {np.max(swing_up_times):7.2f} seconds")
    print()
    print(f"Mean Episode Length:  {np.mean(all_lengths):7.1f} steps ({np.mean(all_lengths)*0.005:.1f}s)")
    print(f"Mean Episode Reward:  {np.mean(all_rewards):7.1f}")
    print(f"{'='*70}\n")

    return {
        'all_lengths': np.array(all_lengths),
        'all_rewards': np.array(all_rewards),
        'swing_up_times': np.array(swing_up_times) if swing_up_times else np.array([]),
        'successful_episodes': successful_episodes,
        'success_rate': success_rate,
        'swing_up_success_rate': swing_up_success_rate,
        'mean_swing_up_time': mean_swing_up_time,
        'n_episodes': n_episodes
    }


def plot_results(stats, save_dir):
    """Generate evaluation plots."""
    os.makedirs(save_dir, exist_ok=True)

    # Plot 1: Success vs Failure
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left plot: Full success (swing-up AND stabilization)
    ax = axes[0]
    successful_episodes = stats['successful_episodes']
    n_episodes = stats['n_episodes']
    failed_episodes = n_episodes - successful_episodes

    categories = ['Success', 'Failure']
    counts = [successful_episodes, failed_episodes]
    colors = ['#2ecc71', '#e74c3c']

    bars = ax.bar(categories, counts, color=colors, alpha=0.8, edgecolor='black', linewidth=2)

    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}\n({count/n_episodes*100:.1f}%)',
                ha='center', va='bottom', fontsize=14, fontweight='bold')

    ax.set_ylabel('Number of Episodes', fontsize=12, fontweight='bold')
    ax.set_title(f'Full Task Success: Swing-Up + Stabilization\n({n_episodes} episodes)',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_ylim(0, n_episodes + 10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Right plot: Swing-up time distribution
    ax = axes[1]
    if len(stats['swing_up_times']) > 0:
        ax.hist(stats['swing_up_times'], bins=20, color='#3498db', alpha=0.7, edgecolor='black')
        ax.axvline(stats['mean_swing_up_time'], color='red', linestyle='--',
                   linewidth=2, label=f"Mean: {stats['mean_swing_up_time']:.2f}s")
        ax.set_xlabel('Time to Reach Upright (seconds)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title(f'Swing-Up Time Distribution\n({len(stats["swing_up_times"])}/{n_episodes} successful)',
                     fontsize=14, fontweight='bold', pad=15)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    else:
        ax.text(0.5, 0.5, 'No successful swing-ups', ha='center', va='center',
                fontsize=16, transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    summary_path = os.path.join(save_dir, 'swingup_evaluation.png')
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    print(f"Evaluation plot saved to: {summary_path}")
    plt.close()


if __name__ == '__main__':
    # Get project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # Model path from training
    model_path = os.path.join(project_root, 'results/ppo_swingup/best_model/best_model')
    save_dir = os.path.join(project_root, 'results/ppo_swingup')

    # Run evaluation
    stats = evaluate_model(model_path, n_episodes=100, max_episode_steps=10000)

    # Generate plots
    plot_results(stats, save_dir)

    print("\n" + "="*70)
    print("Swing-up evaluation complete!")
    print(f"Results saved to: {save_dir}")
    print("="*70)

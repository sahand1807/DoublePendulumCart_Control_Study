"""
Evaluate trained PPO Level 2 model and generate plots.
Tests the model on ±6° perturbations and visualizes performance.
"""
# Add parent directory to path for imports when running from experiments/
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from env.double_pendulum_cart_env import DoublePendulumCartEnv
from env.angle_wrapper import AngleObservationWrapper, CurriculumInitializationWrapper

def create_level2_env():
    """Create Level 2 environment with ±6° perturbations."""
    env = DoublePendulumCartEnv()
    env = AngleObservationWrapper(env)
    env = CurriculumInitializationWrapper(env, curriculum_level=2)
    return env

def evaluate_model(model_path, n_episodes=100):
    """Evaluate model and collect episode lengths."""
    print(f"\n{'='*60}")
    print(f"Loading model: {model_path}")
    print(f"Testing on {n_episodes} episodes...")
    print(f"{'='*60}\n")

    model = PPO.load(model_path)
    env = create_level2_env()

    # Storage for metrics
    all_lengths = []
    successful_episodes = 0

    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_length = 0

        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_length += 1

        all_lengths.append(episode_length)

        # Success: episode lasted ≥1900 steps (95% of 2000)
        if episode_length >= 1900:
            successful_episodes += 1

        # Print progress every 10 episodes
        if (episode + 1) % 10 == 0:
            print(f"Progress: {episode+1}/{n_episodes} episodes completed")

    env.close()

    # Print summary statistics
    print(f"\n{'='*60}")
    print(f"EVALUATION SUMMARY (n={n_episodes} episodes)")
    print(f"{'='*60}")
    print(f"Success Rate:         {100*successful_episodes/n_episodes:6.1f}% ({successful_episodes}/{n_episodes})")
    print(f"Mean Episode Length:  {np.mean(all_lengths):7.1f} steps")
    print(f"Median Episode Length: {np.median(all_lengths):7.1f} steps")
    print(f"Min Episode Length:   {np.min(all_lengths):7d} steps")
    print(f"Max Episode Length:   {np.max(all_lengths):7d} steps")
    print(f"{'='*60}\n")

    return {
        'all_lengths': np.array(all_lengths),
        'successful_episodes': successful_episodes,
        'success_rate': 100*successful_episodes/n_episodes,
        'n_episodes': n_episodes
    }

def plot_results(stats, save_dir):
    """Generate bar plot showing success vs failure counts."""
    os.makedirs(save_dir, exist_ok=True)

    successful_episodes = stats['successful_episodes']
    n_episodes = stats['n_episodes']
    failed_episodes = n_episodes - successful_episodes

    # Create single bar chart
    fig, ax = plt.subplots(figsize=(10, 6))

    categories = ['Success', 'Failure']
    counts = [successful_episodes, failed_episodes]
    colors = ['#2ecc71', '#e74c3c']

    bars = ax.bar(categories, counts, color=colors, alpha=0.8, edgecolor='black', linewidth=2)

    # Add count labels on top of bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}\n({count/n_episodes*100:.1f}%)',
                ha='center', va='bottom', fontsize=16, fontweight='bold')

    ax.set_ylabel('Number of Episodes', fontsize=14, fontweight='bold')
    ax.set_title(f'PPO Level 2 Evaluation: Success vs Failure\n({n_episodes} episodes, 10 seconds each)\nSuccess = Episode Length ≥ 1900 steps',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_ylim(0, n_episodes + 10)
    ax.grid(True, alpha=0.3, axis='y')

    # Make the plot look cleaner
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    summary_path = os.path.join(save_dir, 'level2_evaluation.png')
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    print(f"\nEvaluation plot saved to: {summary_path}")
    plt.close()

if __name__ == '__main__':
    # Add parent directory to path for imports when running from experiments/
    import os
    import sys

    # Get project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    sys.path.insert(0, project_root)

    # Model path from training (absolute path from project root)
    # Note: PPO.load() automatically adds .zip extension
    model_path = os.path.join(project_root, 'results/ppo_level2/best_model/best_model')
    save_dir = os.path.join(project_root, 'results/ppo_level2')

    # Run evaluation
    stats = evaluate_model(model_path, n_episodes=100)

    # Generate plots
    plot_results(stats, save_dir)

    print("\n" + "="*60)
    print("Evaluation complete!")
    print(f"Results saved to: {save_dir}")
    print("="*60)

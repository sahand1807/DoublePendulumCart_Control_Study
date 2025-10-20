"""
Plot learning curves for swing-up training.

This script generates comprehensive visualizations of the swing-up training progress,
including:
- Episode reward over time
- Episode length over time (indicates swing-up success)
- Policy entropy (exploration metric)
- Value and policy losses
- Explained variance

Usage:
    python experiments/plot_swingup_curves.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.plot_learning_curves import plot_learning_curves


def main():
    # Get project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # Paths for swing-up training logs
    log_dir = os.path.join(project_root, 'results/ppo_swingup/PPO_1')
    save_dir = os.path.join(project_root, 'results/ppo_swingup')

    print("\n" + "="*70)
    print("SWING-UP TRAINING - LEARNING CURVES")
    print("="*70)
    print(f"Log directory: {log_dir}")
    print(f"Save directory: {save_dir}")
    print("="*70 + "\n")

    # Generate learning curves
    # Smoothing window=20 for swing-up (noisier than stabilization)
    plot_learning_curves(log_dir, save_dir, window=20)

    print("\n" + "="*70)
    print("Learning curves generated successfully!")
    print("="*70)
    print("\nGenerated files:")
    print("  - learning_curves.png: Comprehensive training metrics")
    print("  - reward_curve_hires.png: High-resolution reward plot with confidence interval")
    print("\nKey metrics to look for:")
    print("  - Episode Reward: Should increase and stabilize")
    print("  - Episode Length: Should increase (longer = better stabilization)")
    print("  - Entropy: Should decrease as policy becomes more deterministic")
    print("  - Explained Variance: Should approach 1.0 (good value function)")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()

"""
Generate publication-quality learning curves for RL training.

Standard RL plots include:
1. Episode reward vs timesteps (with confidence intervals)
2. Episode length vs timesteps
3. Success rate vs timesteps
4. Value loss and policy loss
5. Learning rate schedule
6. Entropy (exploration metric)
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import json
from stable_baselines3.common.monitor import LoadMonitorResultsError
from stable_baselines3.common.results_plotter import load_results, ts2xy
from tensorboard.backend.event_processing import event_accumulator


def smooth_curve(data, window=10):
    """Apply moving average smoothing to noisy data."""
    if len(data) < window:
        return data
    cumsum = np.cumsum(np.insert(data, 0, 0))
    return (cumsum[window:] - cumsum[:-window]) / window


def load_tensorboard_logs(log_dir):
    """
    Load training metrics from TensorBoard event files.

    Args:
        log_dir: Directory containing TensorBoard event files

    Returns:
        DataFrame with training metrics
    """
    log_dir = Path(log_dir)

    # Find all event files
    event_files = list(log_dir.glob('**/events.out.tfevents.*'))

    if not event_files:
        print(f"Warning: No TensorBoard event files found in {log_dir}")
        return None

    print(f"Found {len(event_files)} TensorBoard event file(s)")

    # Load data from all event files
    all_data = {}

    for event_file in event_files:
        print(f"Loading: {event_file.name}")

        # Create event accumulator
        ea = event_accumulator.EventAccumulator(
            str(event_file),
            size_guidance={
                event_accumulator.SCALARS: 0,  # Load all scalars
            }
        )
        ea.Reload()

        # Get all scalar tags
        tags = ea.Tags()['scalars']
        print(f"  Found {len(tags)} metrics: {tags[:5]}..." if len(tags) > 5 else f"  Found {len(tags)} metrics")

        # Extract each scalar
        for tag in tags:
            events = ea.Scalars(tag)
            steps = [e.step for e in events]
            values = [e.value for e in events]

            if tag not in all_data:
                all_data[tag] = {'steps': [], 'values': []}

            all_data[tag]['steps'].extend(steps)
            all_data[tag]['values'].extend(values)

    # Convert to DataFrame
    if not all_data:
        print("Warning: No data loaded from TensorBoard files")
        return None

    # Create a unified dataframe
    # Find the common tag to use as index (usually rollout/ep_rew_mean)
    dfs = []
    for tag, data in all_data.items():
        df_tag = pd.DataFrame({
            'step': data['steps'],
            tag: data['values']
        })
        # Sort by step and remove duplicates
        df_tag = df_tag.sort_values('step').drop_duplicates('step')
        dfs.append(df_tag)

    # Merge all dataframes on step
    if dfs:
        df_merged = dfs[0]
        for df in dfs[1:]:
            df_merged = pd.merge(df_merged, df, on='step', how='outer')
        df_merged = df_merged.sort_values('step').reset_index(drop=True)

        print(f"\nLoaded data shape: {df_merged.shape}")
        print(f"Columns: {list(df_merged.columns)}")
        return df_merged

    return None


def plot_learning_curves(log_dir, save_dir=None, window=10):
    """
    Generate comprehensive learning curves from training logs.

    Args:
        log_dir: Directory containing TensorBoard logs or CSV files
        save_dir: Where to save plots (defaults to log_dir)
        window: Smoothing window size
    """
    log_dir = Path(log_dir)
    save_dir = Path(save_dir) if save_dir else log_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating learning curves from: {log_dir}")
    print(f"Saving plots to: {save_dir}")

    # Try to load TensorBoard data first
    df_tensorboard = load_tensorboard_logs(log_dir)
    has_tensorboard = df_tensorboard is not None

    # Load monitor data (episode-level metrics)
    try:
        df_monitor = load_results(str(log_dir))
        has_monitor = True
    except (LoadMonitorResultsError, FileNotFoundError):
        print("Warning: No monitor.csv found, skipping episode-based plots")
        has_monitor = False

    # Load progress.csv data (training metrics) - fallback if no TensorBoard
    progress_file = log_dir / "progress.csv"
    if progress_file.exists():
        df_progress = pd.read_csv(progress_file)
        has_progress = True
    else:
        has_progress = False
        if not has_tensorboard:
            print("Warning: No progress.csv or TensorBoard data found")

    # Use TensorBoard data if available, otherwise use progress.csv
    if has_tensorboard:
        df_progress = df_tensorboard
        has_progress = True
        print("Using TensorBoard data for training metrics")

    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # ========== Plot 1: Episode Reward (from TensorBoard or monitor) ==========
    ax1 = fig.add_subplot(gs[0, 0])
    if has_monitor:
        timesteps, rewards = ts2xy(df_monitor, 'timesteps')

        # Raw data (light)
        ax1.plot(timesteps, rewards, alpha=0.2, color='tab:blue', linewidth=0.5)

        # Smoothed data
        if len(rewards) > window:
            smoothed_rewards = smooth_curve(rewards, window)
            smooth_timesteps = timesteps[window-1:]
            ax1.plot(smooth_timesteps, smoothed_rewards, color='tab:blue',
                    linewidth=2, label=f'Smoothed (window={window})')
    elif has_progress and 'rollout/ep_rew_mean' in df_progress.columns:
        # Use TensorBoard mean reward
        timesteps = df_progress['step'].values
        ep_rew_mean = df_progress['rollout/ep_rew_mean'].values
        # Remove NaN values
        mask = ~np.isnan(ep_rew_mean)
        ax1.plot(timesteps[mask], ep_rew_mean[mask], color='tab:blue', linewidth=2, marker='o', markersize=3)

    ax1.set_xlabel('Timesteps', fontsize=12)
    ax1.set_ylabel('Episode Reward', fontsize=12)
    ax1.set_title('Episode Reward vs Timesteps', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # ========== Plot 2: Episode Length ==========
    ax2 = fig.add_subplot(gs[0, 1])
    if has_monitor:
        timesteps, lengths = ts2xy(df_monitor, 'timesteps')
        episode_lengths = df_monitor['l'].values

        # Raw data
        ax2.plot(timesteps, episode_lengths, alpha=0.2, color='tab:green', linewidth=0.5)

        # Smoothed data
        if len(episode_lengths) > window:
            smoothed_lengths = smooth_curve(episode_lengths, window)
            smooth_timesteps = timesteps[window-1:]
            ax2.plot(smooth_timesteps, smoothed_lengths, color='tab:green',
                    linewidth=2, label=f'Smoothed (window={window})')
    elif has_progress and 'rollout/ep_len_mean' in df_progress.columns:
        # Use TensorBoard mean episode length
        timesteps = df_progress['step'].values
        ep_len_mean = df_progress['rollout/ep_len_mean'].values
        # Remove NaN values
        mask = ~np.isnan(ep_len_mean)
        ax2.plot(timesteps[mask], ep_len_mean[mask], color='tab:green', linewidth=2, marker='o', markersize=3)

    ax2.set_xlabel('Timesteps', fontsize=12)
    ax2.set_ylabel('Episode Length', fontsize=12)
    ax2.set_title('Episode Length vs Timesteps', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # ========== Plot 3: Entropy (Exploration) ==========
    if has_progress and 'train/entropy_loss' in df_progress.columns:
        ax3 = fig.add_subplot(gs[0, 2])
        timesteps = df_progress.get('time/total_timesteps', df_progress['step']).values
        entropy = -df_progress['train/entropy_loss'].values  # Negate for positive entropy
        mask = ~np.isnan(entropy)

        ax3.plot(timesteps[mask], entropy[mask], color='tab:cyan', linewidth=2, marker='o', markersize=3)
        ax3.set_xlabel('Timesteps', fontsize=12)
        ax3.set_ylabel('Policy Entropy', fontsize=12)
        ax3.set_title('Exploration (Entropy) vs Timesteps', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)

    # ========== Plot 4: Value Loss ==========
    if has_progress and 'train/value_loss' in df_progress.columns:
        ax4 = fig.add_subplot(gs[1, 0])
        timesteps = df_progress.get('time/total_timesteps', df_progress['step']).values
        value_loss = df_progress['train/value_loss'].values
        mask = ~np.isnan(value_loss)

        ax4.plot(timesteps[mask], value_loss[mask], color='tab:red', linewidth=2, marker='o', markersize=3)
        ax4.set_xlabel('Timesteps', fontsize=12)
        ax4.set_ylabel('Value Loss', fontsize=12)
        ax4.set_title('Critic Loss vs Timesteps', fontsize=14, fontweight='bold')
        ax4.set_yscale('log')
        ax4.grid(True, alpha=0.3)

    # ========== Plot 5: Policy Gradient Loss ==========
    if has_progress and 'train/policy_gradient_loss' in df_progress.columns:
        ax5 = fig.add_subplot(gs[1, 1])
        timesteps = df_progress.get('time/total_timesteps', df_progress['step']).values
        pg_loss = df_progress['train/policy_gradient_loss'].values
        mask = ~np.isnan(pg_loss)

        ax5.plot(timesteps[mask], pg_loss[mask], color='tab:purple', linewidth=2, marker='o', markersize=3)
        ax5.set_xlabel('Timesteps', fontsize=12)
        ax5.set_ylabel('Policy Gradient Loss', fontsize=12)
        ax5.set_title('Actor Loss vs Timesteps', fontsize=14, fontweight='bold')
        ax5.grid(True, alpha=0.3)

    # ========== Plot 6: Explained Variance ==========
    if has_progress and 'train/explained_variance' in df_progress.columns:
        ax6 = fig.add_subplot(gs[1, 2])
        timesteps = df_progress.get('time/total_timesteps', df_progress['step']).values
        expl_var = df_progress['train/explained_variance'].values
        mask = ~np.isnan(expl_var)

        ax6.plot(timesteps[mask], expl_var[mask], color='tab:pink', linewidth=2, marker='o', markersize=3)
        ax6.axhline(y=1.0, color='gray', linestyle='-', linewidth=1, alpha=0.5, label='Perfect')
        ax6.set_xlabel('Timesteps', fontsize=12)
        ax6.set_ylabel('Explained Variance', fontsize=12)
        ax6.set_title('Value Function Quality', fontsize=14, fontweight='bold')
        ax6.set_ylim(0, 1.1)
        ax6.legend()
        ax6.grid(True, alpha=0.3)

    # Plot 7, 8, 9 removed - keeping only 6 essential plots

    # Add overall title
    fig.suptitle('PPO Training Learning Curves', fontsize=16, fontweight='bold', y=0.995)

    # Save figure
    save_path = save_dir / 'learning_curves.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved comprehensive learning curves to: {save_path}")

    plt.close()

    # ========== Create separate high-resolution reward plot ==========
    if has_monitor:
        fig_reward = plt.figure(figsize=(12, 6))
        ax = fig_reward.add_subplot(111)

        timesteps, rewards = ts2xy(df_monitor, 'timesteps')

        # Plot with confidence interval (if enough data)
        if len(rewards) > window * 2:
            # Compute rolling mean and std
            smoothed_rewards = smooth_curve(rewards, window)
            smooth_timesteps = timesteps[window-1:]

            # Compute rolling std for confidence interval
            rolling_std = pd.Series(rewards).rolling(window=window).std().values[window-1:]

            # Plot
            ax.fill_between(smooth_timesteps,
                           smoothed_rewards - rolling_std,
                           smoothed_rewards + rolling_std,
                           alpha=0.2, color='tab:blue', label='±1 std')
            ax.plot(smooth_timesteps, smoothed_rewards, color='tab:blue',
                   linewidth=3, label='Mean reward')
        else:
            ax.plot(timesteps, rewards, color='tab:blue', linewidth=2)

        ax.set_xlabel('Timesteps', fontsize=14)
        ax.set_ylabel('Episode Reward', fontsize=14)
        ax.set_title('Episode Reward with Confidence Interval', fontsize=16, fontweight='bold')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)

        save_path_reward = save_dir / 'reward_curve_hires.png'
        plt.savefig(save_path_reward, dpi=300, bbox_inches='tight')
        print(f"Saved high-resolution reward curve to: {save_path_reward}")

        plt.close()

    # ========== Print training statistics ==========
    if has_monitor:
        print("\n" + "="*60)
        print("TRAINING STATISTICS")
        print("="*60)
        print(f"Total episodes: {len(df_monitor)}")
        print(f"Total timesteps: {df_monitor['t'].max():.0f}")
        print(f"\nFinal 100 episodes:")
        final_rewards = df_monitor['r'].values[-100:]
        final_lengths = df_monitor['l'].values[-100:]
        print(f"  Mean reward: {final_rewards.mean():.2f} ± {final_rewards.std():.2f}")
        print(f"  Mean length: {final_lengths.mean():.0f} ± {final_lengths.std():.0f}")
        print(f"  Max reward: {final_rewards.max():.2f}")
        print(f"  Min reward: {final_rewards.min():.2f}")
        print("="*60 + "\n")

    return save_dir


def plot_multi_run_comparison(log_dirs, labels, save_dir, window=10):
    """
    Compare multiple training runs on the same plot.

    Args:
        log_dirs: List of directories containing training logs
        labels: List of labels for each run
        save_dir: Where to save comparison plot
        window: Smoothing window
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(log_dirs)))

    for log_dir, label, color in zip(log_dirs, labels, colors):
        try:
            df = load_results(str(log_dir))
            timesteps, rewards = ts2xy(df, 'timesteps')

            if len(rewards) > window:
                smoothed_rewards = smooth_curve(rewards, window)
                smooth_timesteps = timesteps[window-1:]

                # Reward plot
                ax1.plot(smooth_timesteps, smoothed_rewards, color=color,
                        linewidth=2, label=label, alpha=0.8)

                # Episode length plot
                lengths = df['l'].values
                smoothed_lengths = smooth_curve(lengths, window)
                ax2.plot(smooth_timesteps, smoothed_lengths, color=color,
                        linewidth=2, label=label, alpha=0.8)
        except Exception as e:
            print(f"Warning: Could not load {log_dir}: {e}")

    ax1.set_xlabel('Timesteps', fontsize=12)
    ax1.set_ylabel('Episode Reward', fontsize=12)
    ax1.set_title('Reward Comparison', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Timesteps', fontsize=12)
    ax2.set_ylabel('Episode Length', fontsize=12)
    ax2.set_title('Episode Length Comparison', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = save_dir / 'multi_run_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved multi-run comparison to: {save_path}")

    plt.close()


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python plot_learning_curves.py <log_dir> [save_dir] [window]")
        sys.exit(1)

    log_dir = sys.argv[1]
    save_dir = sys.argv[2] if len(sys.argv) > 2 else None
    window = int(sys.argv[3]) if len(sys.argv) > 3 else 10

    plot_learning_curves(log_dir, save_dir, window)

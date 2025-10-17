"""
Robustness Study: Initial Condition Sweep

Maps the region of attraction by testing a grid of initial angles.
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.double_pendulum_cart_env import DoublePendulumCartEnv
from controllers.lqr_controller import create_lqr_controller


def run_initial_condition_sweep():
    """
    Test LQR controller across grid of initial conditions.
    Map the region of attraction.
    """
    print("="*80)
    print("ROBUSTNESS STUDY: INITIAL CONDITION SWEEP")
    print("="*80)
    
    # Create results directory
    results_dir = "results/analysis/lqr"
    os.makedirs(results_dir, exist_ok=True)
    
    # Create baseline controller
    controller = create_lqr_controller()
    
    # Define grid of initial angles
    theta1_range = np.arange(-30, 31, 5)  # -30° to +30° in 5° steps
    theta2_range = np.arange(-30, 31, 5)  # -30° to +30° in 5° steps
    
    print(f"\nTesting grid: θ₁ ∈ [{theta1_range[0]}°, {theta1_range[-1]}°], θ₂ ∈ [{theta2_range[0]}°, {theta2_range[-1]}°]")
    print(f"Grid size: {len(theta1_range)} × {len(theta2_range)} = {len(theta1_range)*len(theta2_range)} tests")
    print(f"Simulation: 5 seconds (500 steps)")
    
    # Success criteria
    print("\nSuccess Criteria:")
    print("  1. Both angles stay within ±60° from upright throughout simulation")
    print("  2. Both angles settle within ±3° for at least 1 second")
    print("  3. No early termination")
    
    # Storage
    success_grid = np.zeros((len(theta1_range), len(theta2_range)))
    settling_times = np.full((len(theta1_range), len(theta2_range)), np.nan)
    max_deviations = np.zeros((len(theta1_range), len(theta2_range)))
    
    total_tests = len(theta1_range) * len(theta2_range)
    test_count = 0
    success_count = 0
    
    print("\n" + "-"*80)
    print("Running tests...")
    
    for i, theta1_deg in enumerate(theta1_range):
        for j, theta2_deg in enumerate(theta2_range):
            test_count += 1
            
            # Create initial state
            initial_state = np.array([
                0.0,                                    # x = 0
                np.pi + np.radians(theta1_deg),        # θ₁
                np.pi + np.radians(theta2_deg),        # θ₂
                0.0, 0.0, 0.0                          # All velocities = 0
            ])
            
            # Run test
            result = test_initial_condition(controller, initial_state)
            
            # Store results
            success_grid[i, j] = 1 if result['success'] else 0
            if result['success']:
                settling_times[i, j] = result['settling_time']
                success_count += 1
            max_deviations[i, j] = result['max_deviation']
            
            # Progress update
            if test_count % 20 == 0:
                success_rate = (success_count / test_count) * 100
                print(f"  Progress: {test_count}/{total_tests} ({test_count*100//total_tests}%) | Success rate: {success_rate:.1f}%")
    
    # Final statistics
    success_rate = (success_count / total_tests) * 100
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"Total tests: {total_tests}")
    print(f"Successful: {success_count} ({success_rate:.1f}%)")
    print(f"Failed: {total_tests - success_count} ({100-success_rate:.1f}%)")
    
    if success_count > 0:
        valid_settling_times = settling_times[~np.isnan(settling_times)]
        print(f"\nSettling times (successful cases):")
        print(f"  Mean: {np.mean(valid_settling_times):.2f}s")
        print(f"  Min: {np.min(valid_settling_times):.2f}s")
        print(f"  Max: {np.max(valid_settling_times):.2f}s")
    
    print("="*80)
    
    # Create plots
    print("\nGenerating plots...")
    create_initial_condition_plots(theta1_range, theta2_range, success_grid, 
                                   settling_times, max_deviations, results_dir)
    
    # Save data
    results = {
        'theta1_range': theta1_range,
        'theta2_range': theta2_range,
        'success_grid': success_grid,
        'settling_times': settling_times,
        'max_deviations': max_deviations,
        'success_rate': success_rate,
        'success_count': success_count,
        'total_tests': total_tests,
    }
    
    return results


def test_initial_condition(controller, initial_state, max_steps=500):
    """
    Test a single initial condition.
    
    Returns:
        dict with 'success', 'settling_time', 'max_deviation', 'failure_reason'
    """
    env = DoublePendulumCartEnv(terminate_on_fall=False)
    controller.reset()
    
    obs, _ = env.reset(options={"initial_state": initial_state})
    
    states = [obs.copy()]
    
    failure_reason = None
    
    for step in range(max_steps):
        u = controller.compute_control(obs)
        action = np.clip(u / 20.0, -1.0, 1.0)
        obs, reward, terminated, truncated, info = env.step([action])
        
        states.append(obs.copy())
        
        # Check failure condition: angles exceed ±60° from upright
        theta1_dev = abs(obs[1] - np.pi)
        theta2_dev = abs(obs[2] - np.pi)
        
        if theta1_dev > np.radians(60) or theta2_dev > np.radians(60):
            failure_reason = "angle_limit_exceeded"
            break
        
        if terminated or truncated:
            failure_reason = "early_termination"
            break
    
    states = np.array(states)
    dt = env.dt * env.frame_skip
    
    # Compute metrics
    max_theta1_dev = np.max(np.abs(states[:, 1] - np.pi))
    max_theta2_dev = np.max(np.abs(states[:, 2] - np.pi))
    max_deviation = np.degrees(max(max_theta1_dev, max_theta2_dev))
    
    settling_time = compute_settling_time(states, dt)
    
    # Success criteria
    success = (failure_reason is None) and (settling_time < max_steps * dt)
    
    return {
        'success': success,
        'settling_time': settling_time,
        'max_deviation': max_deviation,
        'failure_reason': failure_reason,
        'num_steps': len(states) - 1,
    }


def compute_settling_time(states, dt, threshold_deg=3.0):
    """Compute settling time."""
    threshold_rad = np.radians(threshold_deg)
    
    for i in range(len(states)):
        if i < len(states) - 100:  # Need at least 100 steps (1 second) ahead
            future_theta1 = np.abs(states[i:i+100, 1] - np.pi)
            future_theta2 = np.abs(states[i:i+100, 2] - np.pi)
            
            if np.all(future_theta1 < threshold_rad) and np.all(future_theta2 < threshold_rad):
                return i * dt
    
    return len(states) * dt  # Did not settle


def create_initial_condition_plots(theta1_range, theta2_range, success_grid, 
                                   settling_times, max_deviations, save_dir):
    """Create heatmaps for initial condition study."""
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Region of Attraction Analysis', fontsize=16, fontweight='bold')
    
    # 1. Success/Failure Heatmap
    ax1 = axes[0]
    im1 = ax1.imshow(success_grid.T, origin='lower', cmap='RdYlGn', 
                     extent=[theta1_range[0], theta1_range[-1], 
                            theta2_range[0], theta2_range[-1]],
                     aspect='auto', vmin=0, vmax=1)
    ax1.set_xlabel('θ₁ Initial Error (degrees)', fontsize=11)
    ax1.set_ylabel('θ₂ Initial Error (degrees)', fontsize=11)
    ax1.set_title('Region of Attraction (Green = Success)', fontweight='bold')
    ax1.grid(True, alpha=0.3, color='white', linewidth=0.5)
    
    # Add success rate text
    success_rate = np.sum(success_grid) / success_grid.size * 100
    ax1.text(0.95, 0.05, f'Success Rate: {success_rate:.1f}%', 
            transform=ax1.transAxes, fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            verticalalignment='bottom', horizontalalignment='right')
    
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('Success (1) / Failure (0)', fontsize=10)
    
    # 2. Settling Time Heatmap (only for successful cases)
    ax2 = axes[1]
    im2 = ax2.imshow(settling_times.T, origin='lower', cmap='viridis', 
                     extent=[theta1_range[0], theta1_range[-1], 
                            theta2_range[0], theta2_range[-1]],
                     aspect='auto')
    ax2.set_xlabel('θ₁ Initial Error (degrees)', fontsize=11)
    ax2.set_ylabel('θ₂ Initial Error (degrees)', fontsize=11)
    ax2.set_title('Settling Time (Successful Cases)', fontweight='bold')
    ax2.grid(True, alpha=0.3, color='white', linewidth=0.5)
    
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('Settling Time (s)', fontsize=10)
    
    # 3. Maximum Deviation Heatmap
    ax3 = axes[2]
    im3 = ax3.imshow(max_deviations.T, origin='lower', cmap='hot_r', 
                     extent=[theta1_range[0], theta1_range[-1], 
                            theta2_range[0], theta2_range[-1]],
                     aspect='auto')
    ax3.set_xlabel('θ₁ Initial Error (degrees)', fontsize=11)
    ax3.set_ylabel('θ₂ Initial Error (degrees)', fontsize=11)
    ax3.set_title('Maximum Angular Deviation', fontweight='bold')
    ax3.grid(True, alpha=0.3, color='white', linewidth=0.5)
    
    # Add 60° failure threshold line
    ax3.contour(theta1_range, theta2_range, max_deviations.T, 
               levels=[60], colors='red', linewidths=2, linestyles='--')
    ax3.text(0.95, 0.95, '60° failure threshold', 
            transform=ax3.transAxes, fontsize=10, color='red',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            verticalalignment='top', horizontalalignment='right')
    
    cbar3 = plt.colorbar(im3, ax=ax3)
    cbar3.set_label('Max Deviation (degrees)', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'initial_condition_region_of_attraction.png'), 
                dpi=200, bbox_inches='tight')
    print(f"  ✓ Saved: initial_condition_region_of_attraction.png")
    plt.close()


if __name__ == "__main__":
    results = run_initial_condition_sweep()
    print("\n✅ Initial condition sweep complete!")
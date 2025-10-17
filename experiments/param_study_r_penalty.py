"""
Parameter Study: R-Matrix Control Penalty Variation

Investigates the trade-off between performance and control effort.
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.double_pendulum_cart_env import DoublePendulumCartEnv
from controllers.lqr_controller import create_lqr_controller


def run_r_penalty_study():
    """
    Study effect of control penalty R on LQR performance.
    Focus: Trade-off between settling time and control effort.
    """
    print("="*80)
    print("PARAMETER STUDY: CONTROL PENALTY (R) VARIATION")
    print("="*80)
    
    # Create results directory
    results_dir = "results/analysis/lqr"
    os.makedirs(results_dir, exist_ok=True)
    
    # Test different R values (control penalty)
    r_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    
    # Fixed Q weights (baseline)
    q_x = 1.0
    q_angle = 100.0
    q_vel = 0.1
    q_ang_vel = 10.0
    
    # Initial condition (same as angle weight study)
    initial_state = np.array([
        0.0,                          # x = 0m
        np.pi + np.radians(10),       # θ₁ = 180° + 10°
        np.pi + np.radians(8),        # θ₂ = 180° + 8°
        0.0, 0.0, 0.0                 # All velocities = 0
    ])
    max_steps = 600
    
    # Storage
    results = []
    
    print(f"\nTesting R values: {r_values}")
    print(f"Fixed: Q_x={q_x}, Q_θ={q_angle}, Q_vel={q_vel}, Q_ang_vel={q_ang_vel}")
    print(f"Initial: θ₁={np.degrees(initial_state[1])-180:.1f}°, θ₂={np.degrees(initial_state[2])-180:.1f}°")
    print("\n" + "-"*80)
    
    for r_val in r_values:
        print(f"\nTesting R = {r_val}...")
        
        # Create Q and R matrices
        Q = np.diag([q_x, q_angle, q_angle, q_vel, q_ang_vel, q_ang_vel])
        R = np.array([[r_val]])
        
        # Create controller
        controller = create_lqr_controller(Q=Q, R=R)
        
        # Create environment
        env = DoublePendulumCartEnv(terminate_on_fall=False)
        
        # Run experiment
        obs, _ = env.reset(options={"initial_state": initial_state})
        states = [obs.copy()]
        actions = []
        
        for step in range(max_steps):
            u = controller.compute_control(obs)
            action = np.clip(u / 20.0, -1.0, 1.0)
            obs, reward, terminated, truncated, info = env.step([action])
            
            states.append(obs.copy())
            actions.append(u)
            
            if terminated or truncated:
                break
        
        states = np.array(states)
        actions = np.array(actions)
        
        # Compute metrics
        dt = env.dt * env.frame_skip
        settling_time = compute_settling_time(states, dt)
        control_effort = np.sum(actions**2)
        peak_force = np.max(np.abs(actions))
        saturation_count = np.sum(np.abs(actions) > 20.0)
        saturation_percent = (saturation_count / len(actions)) * 100 if len(actions) > 0 else 0.0
        final_error = np.abs(states[-1, 1] - np.pi) + np.abs(states[-1, 2] - np.pi)
        
        # Average force (normalized energy metric)
        rms_force = np.sqrt(np.mean(actions**2))
        
        results.append({
            'r_value': r_val,
            'settling_time': settling_time,
            'control_effort': control_effort,
            'peak_force': peak_force,
            'rms_force': rms_force,
            'saturation_percent': saturation_percent,
            'final_error': final_error,
            'states': states,
            'actions': actions,
            'dt': dt,
        })
        
        print(f"  Settling: {settling_time:.2f}s | Effort: {control_effort:.1f} N²⋅s | "
              f"Peak: {peak_force:.2f}N | RMS: {rms_force:.2f}N | Saturation: {saturation_percent:.1f}%")
    
    # Create plots
    print("\nGenerating plots...")
    create_r_penalty_plots(results, results_dir)
    
    # Print summary table
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"{'R':<8} {'Settling (s)':<14} {'Effort (N²⋅s)':<16} {'Peak (N)':<10} {'RMS (N)':<10} {'Sat (%)':<10}")
    print("-"*80)
    
    for result in results:
        print(f"{result['r_value']:<8} "
              f"{result['settling_time']:<14.2f} "
              f"{result['control_effort']:<16.1f} "
              f"{result['peak_force']:<10.2f} "
              f"{result['rms_force']:<10.2f} "
              f"{result['saturation_percent']:<10.1f}")
    
    print("="*80)
    
    # Generate markdown table
    print("\n" + "="*80)
    print("MARKDOWN TABLE (for report.md Section 4.2):")
    print("="*80)
    print("| R | Settling Time (s) | Control Effort (N²⋅s) | Peak Force (N) | RMS Force (N) | Saturation (%) |")
    print("|---|-------------------|----------------------|----------------|---------------|----------------|")
    
    for result in results:
        print(f"| {result['r_value']} "
              f"| {result['settling_time']:.2f} "
              f"| {result['control_effort']:.1f} "
              f"| {result['peak_force']:.2f} "
              f"| {result['rms_force']:.2f} "
              f"| {result['saturation_percent']:.1f} |")
    
    print("\n✓ Plots saved to: " + results_dir)
    
    return results


def compute_settling_time(states, dt, threshold_deg=3.0):
    """Compute settling time for angles."""
    threshold_rad = np.radians(threshold_deg)
    
    for i in range(len(states)):
        if i < len(states) - 20:
            future_theta1 = np.abs(states[i:i+20, 1] - np.pi)
            future_theta2 = np.abs(states[i:i+20, 2] - np.pi)
            
            if np.all(future_theta1 < threshold_rad) and np.all(future_theta2 < threshold_rad):
                return i * dt
    
    return len(states) * dt


def create_r_penalty_plots(results, save_dir):
    """Create comprehensive plots for R penalty study."""
    
    r_values = [r['r_value'] for r in results]
    settling_times = [r['settling_time'] for r in results]
    control_efforts = [r['control_effort'] for r in results]
    peak_forces = [r['peak_force'] for r in results]
    rms_forces = [r['rms_force'] for r in results]
    saturations = [r['saturation_percent'] for r in results]
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Control Penalty (R) Sensitivity Study', fontsize=16, fontweight='bold')
    
    # 1. Settling Time vs R
    ax1 = axes[0, 0]
    ax1.plot(r_values, settling_times, 'o-', linewidth=2, markersize=8, color='steelblue')
    ax1.set_xlabel('Control Penalty R', fontsize=11)
    ax1.set_ylabel('Settling Time (s)', fontsize=11)
    ax1.set_title('Settling Time vs Control Penalty', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    
    # 2. Control Effort vs R
    ax2 = axes[0, 1]
    ax2.plot(r_values, control_efforts, 's-', linewidth=2, markersize=8, color='darkgreen')
    ax2.set_xlabel('Control Penalty R', fontsize=11)
    ax2.set_ylabel('Control Effort (N²⋅s)', fontsize=11)
    ax2.set_title('Control Effort vs Penalty', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    
    # 3. Peak and RMS Force vs R
    ax3 = axes[1, 0]
    ax3.plot(r_values, peak_forces, '^-', linewidth=2, markersize=8, color='darkred', label='Peak Force')
    ax3.plot(r_values, rms_forces, 'v-', linewidth=2, markersize=8, color='salmon', label='RMS Force')
    ax3.axhline(y=20, color='red', linestyle='--', alpha=0.5, label='Force Limit')
    ax3.set_xlabel('Control Penalty R', fontsize=11)
    ax3.set_ylabel('Force (N)', fontsize=11)
    ax3.set_title('Force Magnitude vs Penalty', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xscale('log')
    
    # 4. Pareto Frontier: Settling Time vs Control Effort
    ax4 = axes[1, 1]
    ax4.scatter(settling_times, control_efforts, s=100, c=r_values, 
                cmap='viridis', edgecolors='black', linewidth=2, zorder=3)
    
    # Draw line connecting points
    ax4.plot(settling_times, control_efforts, 'k--', alpha=0.3, linewidth=1, zorder=1)
    
    # Annotate each point with R value
    for result in results:
        ax4.annotate(f'R={result["r_value"]}', 
                    (result['settling_time'], result['control_effort']),
                    textcoords="offset points", xytext=(5,5), fontsize=8)
    
    ax4.set_xlabel('Settling Time (s)', fontsize=11)
    ax4.set_ylabel('Control Effort (N²⋅s)', fontsize=11)
    ax4.set_title('Pareto Frontier: Performance vs Energy Trade-off', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    cbar = plt.colorbar(ax4.collections[0], ax=ax4)
    cbar.set_label('R value', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'r_penalty_sensitivity.png'), dpi=200, bbox_inches='tight')
    print(f"  ✓ Saved: r_penalty_sensitivity.png")
    plt.close()
    
    # Create time-domain comparison plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle('Time-Domain Response: Control Aggressiveness Comparison', fontsize=14, fontweight='bold')
    
    # Select subset for clarity: most aggressive, baseline, most conservative
    indices_to_plot = [0, 2, 5]  # R = 0.1, 1.0, 10.0
    colors = ['red', 'green', 'blue']
    labels = ['Aggressive (R=0.1)', 'Baseline (R=1.0)', 'Conservative (R=10.0)']
    
    # Subplot 1: Angle error magnitude
    for idx, color, label in zip(indices_to_plot, colors, labels):
        result = results[idx]
        states = result['states']
        dt = result['dt']
        time = np.arange(len(states)) * dt
        
        # Combined angle error
        angle_error = np.sqrt((np.degrees(states[:, 1]) - 180)**2 + 
                             (np.degrees(states[:, 2]) - 180)**2)
        
        axes[0].plot(time, angle_error, color=color, linewidth=2, 
                    label=label, alpha=0.8)
    
    axes[0].axhline(y=3, color='gray', linestyle=':', alpha=0.5, label='3° threshold')
    axes[0].set_ylabel('Total Angle Error (°)', fontsize=11)
    axes[0].set_title('Combined Angle Error √(θ₁² + θ₂²)', fontweight='bold')
    axes[0].legend(fontsize=9, loc='upper right')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim([0, 5])
    
    # Subplot 2: Control force
    for idx, color, label in zip(indices_to_plot, colors, labels):
        result = results[idx]
        actions = result['actions']
        dt = result['dt']
        time = np.arange(len(actions)) * dt
        
        axes[1].plot(time, actions, color=color, linewidth=2, 
                    label=label, alpha=0.8)
    
    axes[1].axhline(y=20, color='red', linestyle='--', alpha=0.5, label='±20N limit')
    axes[1].axhline(y=-20, color='red', linestyle='--', alpha=0.5)
    axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.5)
    axes[1].set_ylabel('Control Force (N)', fontsize=11)
    axes[1].set_title('Control Input (aggressiveness)', fontweight='bold')
    axes[1].legend(fontsize=9, loc='upper right')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim([0, 5])
    
    # Subplot 3: Cumulative energy (integral of u²)
    for idx, color, label in zip(indices_to_plot, colors, labels):
        result = results[idx]
        actions = result['actions']
        dt = result['dt']
        time = np.arange(len(actions)) * dt
        
        # Cumulative control effort
        cumulative_effort = np.cumsum(actions**2) * dt
        
        axes[2].plot(time, cumulative_effort, color=color, linewidth=2, 
                    label=label, alpha=0.8)
    
    axes[2].set_xlabel('Time (s)', fontsize=11)
    axes[2].set_ylabel('Cumulative Energy (N²⋅s)', fontsize=11)
    axes[2].set_title('Cumulative Control Effort ∫u²dt', fontweight='bold')
    axes[2].legend(fontsize=9, loc='upper left')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlim([0, 5])
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'r_penalty_time_comparison.png'), dpi=200, bbox_inches='tight')
    print(f"  ✓ Saved: r_penalty_time_comparison.png")
    plt.close()


if __name__ == "__main__":
    results = run_r_penalty_study()
    print("\n✅ Control penalty (R) study complete!")
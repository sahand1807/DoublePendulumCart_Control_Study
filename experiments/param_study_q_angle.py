"""
Parameter Study: Q-Matrix Angle Weight Sensitivity

Investigates how varying θ₁ and θ₂ weights affects LQR performance.
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.double_pendulum_cart_env import DoublePendulumCartEnv
from controllers.lqr_controller import create_lqr_controller


def run_angle_weight_study():
    """
    Study effect of angle weights on LQR performance.
    """
    print("="*80)
    print("PARAMETER STUDY: ANGLE WEIGHT SENSITIVITY")
    print("="*80)
    
    # Create results directory
    results_dir = "results/analysis/lqr"
    os.makedirs(results_dir, exist_ok=True)
    
    # Test different angle weights
    q_angle_values = [10, 100, 500, 1000]
    
    # Fixed other weights
    q_x = 1.0
    q_vel = 0.1
    q_ang_vel = 10.0
    r = 1.0
    
    # New initial condition
    initial_state = np.array([
        0.0,                          # x = 0
        np.pi + np.radians(10),       # θ₁ = 180° + 10°
        np.pi + np.radians(8),        # θ₂ = 180° + 8°
        0.0, 0.0, 0.0                 # All velocities = 0
    ])
    max_steps = 500
    
    # Storage
    results = []
    
    print(f"\nTesting angle weights: {q_angle_values}")
    print(f"Fixed: Q_x={q_x}, Q_vel={q_vel}, Q_ang_vel={q_ang_vel}, R={r}")
    print(f"Initial: θ₁={np.degrees(initial_state[1])-180:.1f}°, θ₂={np.degrees(initial_state[2])-180:.1f}°")
    print("\n" + "-"*80)
    
    for q_angle in q_angle_values:
        print(f"\nTesting Q_θ = {q_angle}...")
        
        # Create Q matrix
        Q = np.diag([q_x, q_angle, q_angle, q_vel, q_ang_vel, q_ang_vel])
        R = np.array([[r]])
        
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
        
        # Compute overshoot (max deviation from equilibrium)
        theta1_errors = np.abs(states[:, 1] - np.pi)
        theta2_errors = np.abs(states[:, 2] - np.pi)
        max_overshoot = max(np.max(theta1_errors), np.max(theta2_errors))
        
        results.append({
            'q_angle': q_angle,
            'settling_time': settling_time,
            'control_effort': control_effort,
            'peak_force': peak_force,
            'saturation_percent': saturation_percent,
            'final_error': final_error,
            'max_overshoot': np.degrees(max_overshoot),
            'states': states,
            'actions': actions,
            'dt': dt,
        })
        
        print(f"  Settling: {settling_time:.2f}s | Effort: {control_effort:.1f} | Peak: {peak_force:.2f}N | Overshoot: {np.degrees(max_overshoot):.2f}° | Saturation: {saturation_percent:.1f}%")
    
    # Create plots
    print("\nGenerating plots...")
    create_angle_weight_plots(results, results_dir)
    
    # Print summary table
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"{'Q_θ':<8} {'Settling (s)':<14} {'Effort':<12} {'Peak (N)':<10} {'Sat (%)':<10} {'Overshoot (°)':<14}")
    print("-"*80)
    
    for result in results:
        print(f"{result['q_angle']:<8} "
              f"{result['settling_time']:<14.2f} "
              f"{result['control_effort']:<12.1f} "
              f"{result['peak_force']:<10.2f} "
              f"{result['saturation_percent']:<10.1f} "
              f"{result['max_overshoot']:<14.2f}")
    
    print("="*80)
    
    # Generate markdown table
    print("\n" + "="*80)
    print("MARKDOWN TABLE (for report.md Section 4.1.1):")
    print("="*80)
    print("| Q_θ | Settling Time (s) | Control Effort (N²⋅s) | Peak Force (N) | Saturation (%) | Max Overshoot (°) |")
    print("|-----|-------------------|----------------------|----------------|----------------|-------------------|")
    
    for result in results:
        print(f"| {result['q_angle']} "
              f"| {result['settling_time']:.2f} "
              f"| {result['control_effort']:.1f} "
              f"| {result['peak_force']:.2f} "
              f"| {result['saturation_percent']:.1f} "
              f"| {result['max_overshoot']:.2f} |")
    
    print("\n✓ Plots saved to: " + results_dir)
    
    return results


def compute_settling_time(states, dt, threshold_deg=3.0):
    """Compute settling time."""
    threshold_rad = np.radians(threshold_deg)
    
    for i in range(len(states)):
        if i < len(states) - 20:
            future_theta1 = np.abs(states[i:i+20, 1] - np.pi)
            future_theta2 = np.abs(states[i:i+20, 2] - np.pi)
            
            if np.all(future_theta1 < threshold_rad) and np.all(future_theta2 < threshold_rad):
                return i * dt
    
    return len(states) * dt


def create_angle_weight_plots(results, save_dir):
    """Create comprehensive plots for angle weight study."""
    
    q_angles = [r['q_angle'] for r in results]
    settling_times = [r['settling_time'] for r in results]
    control_efforts = [r['control_effort'] for r in results]
    peak_forces = [r['peak_force'] for r in results]
    overshoots = [r['max_overshoot'] for r in results]
    saturations = [r['saturation_percent'] for r in results]
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Angle Weight Sensitivity Study (Q_θ Variation)', fontsize=16, fontweight='bold')
    
    # 1. Settling Time vs Q_θ
    ax1 = axes[0, 0]
    ax1.plot(q_angles, settling_times, 'o-', linewidth=2, markersize=8, color='steelblue')
    ax1.set_xlabel('Angle Weight Q_θ', fontsize=11)
    ax1.set_ylabel('Settling Time (s)', fontsize=11)
    ax1.set_title('Settling Time vs Angle Weight', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    
    # 2. Control Effort vs Q_θ
    ax2 = axes[0, 1]
    ax2.plot(q_angles, control_efforts, 's-', linewidth=2, markersize=8, color='darkgreen')
    ax2.set_xlabel('Angle Weight Q_θ', fontsize=11)
    ax2.set_ylabel('Control Effort (N²⋅s)', fontsize=11)
    ax2.set_title('Control Effort vs Angle Weight', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    
    # 3. Peak Force vs Q_θ
    ax3 = axes[1, 0]
    ax3.plot(q_angles, peak_forces, '^-', linewidth=2, markersize=8, color='darkred')
    ax3.axhline(y=20, color='red', linestyle='--', alpha=0.5, label='Force Limit (20N)')
    ax3.set_xlabel('Angle Weight Q_θ', fontsize=11)
    ax3.set_ylabel('Peak Force (N)', fontsize=11)
    ax3.set_title('Peak Force vs Angle Weight', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xscale('log')
    
    # 4. Saturation vs Q_θ
    ax4 = axes[1, 1]
    ax4.plot(q_angles, saturations, 'd-', linewidth=2, markersize=8, color='purple')
    ax4.set_xlabel('Angle Weight Q_θ', fontsize=11)
    ax4.set_ylabel('Saturation (%)', fontsize=11)
    ax4.set_title('Control Saturation vs Angle Weight', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'q_angle_sensitivity.png'), dpi=200, bbox_inches='tight')
    print(f"  ✓ Saved: q_angle_sensitivity.png")
    plt.close()
    
    # Create time-domain comparison plot with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle('Time-Domain Response Comparison', fontsize=14, fontweight='bold')
    
    # Plot all 4 Q_θ values
    colors = ['blue', 'green', 'orange', 'red']
    
    # Subplot 1: θ₁ error
    for result, color in zip(results, colors):
        states = result['states']
        dt = result['dt']
        time = np.arange(len(states)) * dt
        
        theta1_error = np.degrees(states[:, 1]) - 180
        
        axes[0].plot(time, theta1_error, color=color, linewidth=2, 
                    label=f"Q_θ={result['q_angle']}", alpha=0.8)
    
    axes[0].axhline(y=3, color='gray', linestyle=':', alpha=0.5, label='±3° threshold')
    axes[0].axhline(y=-3, color='gray', linestyle=':', alpha=0.5)
    axes[0].axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.5)
    axes[0].set_ylabel('θ₁ Error (degrees)', fontsize=11)
    axes[0].set_title('First Link Angle Error', fontweight='bold')
    axes[0].legend(fontsize=9, loc='upper right')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim([0, 5])
    
    # Subplot 2: θ₂ error
    for result, color in zip(results, colors):
        states = result['states']
        dt = result['dt']
        time = np.arange(len(states)) * dt
        
        theta2_error = np.degrees(states[:, 2]) - 180
        
        axes[1].plot(time, theta2_error, color=color, linewidth=2, 
                    label=f"Q_θ={result['q_angle']}", alpha=0.8)
    
    axes[1].axhline(y=3, color='gray', linestyle=':', alpha=0.5, label='±3° threshold')
    axes[1].axhline(y=-3, color='gray', linestyle=':', alpha=0.5)
    axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.5)
    axes[1].set_ylabel('θ₂ Error (degrees)', fontsize=11)
    axes[1].set_title('Second Link Angle Error', fontweight='bold')
    axes[1].legend(fontsize=9, loc='upper right')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim([0, 5])
    
    # Subplot 3: Control effort
    for result, color in zip(results, colors):
        actions = result['actions']
        dt = result['dt']
        time = np.arange(len(actions)) * dt
        
        axes[2].plot(time, actions, color=color, linewidth=2, 
                    label=f"Q_θ={result['q_angle']}", alpha=0.8)
    
    axes[2].axhline(y=20, color='red', linestyle='--', alpha=0.5, label='±20N limit')
    axes[2].axhline(y=-20, color='red', linestyle='--', alpha=0.5)
    axes[2].axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.5)
    axes[2].set_xlabel('Time (s)', fontsize=11)
    axes[2].set_ylabel('Control Force (N)', fontsize=11)
    axes[2].set_title('Control Input', fontweight='bold')
    axes[2].legend(fontsize=9, loc='upper right')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlim([0, 5])
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'q_angle_time_comparison.png'), dpi=200, bbox_inches='tight')
    print(f"  ✓ Saved: q_angle_time_comparison.png")
    plt.close()


if __name__ == "__main__":
    results = run_angle_weight_study()
    print("\n✅ Angle weight sensitivity study complete!")
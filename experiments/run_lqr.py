"""
LQR Controller Experiment Runner

Tests the LQR controller on the double pendulum cart system with various
initial conditions and produces comprehensive visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sys
import os
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.double_pendulum_cart_env import DoublePendulumCartEnv
from controllers.lqr_controller import create_lqr_controller


def run_lqr_experiment(controller, env, initial_state, max_steps=500, name="test"):
    """
    Run a single LQR experiment.
    
    Args:
        controller: LQR controller instance
        env: Gym environment
        initial_state: Initial state [x, θ₁, θ₂, ẋ, θ̇₁, θ̇₂]
        max_steps: Maximum number of steps
        name: Experiment name for logging
    
    Returns:
        Dictionary with trajectory data
    """
    print(f"\n{'='*60}")
    print(f"Running LQR Experiment: {name}")
    print(f"{'='*60}")
    print(f"Initial state: x={initial_state[0]:.3f}, θ₁={np.degrees(initial_state[1]):.1f}°, θ₂={np.degrees(initial_state[2]):.1f}°")
    
    # Reset environment
    obs, _ = env.reset(options={"initial_state": initial_state})
    
    # Storage
    states = [obs.copy()]
    actions = []
    rewards = []
    computation_times = []
    
    # Run episode
    for step in range(max_steps):
        # Compute control
        action, comp_time = controller.compute_control_timed(obs)
        
        # Clip to action space
        action_clipped = np.clip(action, -1.0, 1.0)
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(np.array([action_clipped]))
        
        # Store data
        states.append(obs.copy())
        actions.append(action)
        rewards.append(reward)
        computation_times.append(comp_time)
        controller.log_step(obs, np.array([action]))
        
        # Check termination
        if terminated or truncated:
            print(f"Episode ended at step {step+1}")
            break
        
        # Progress
        if (step + 1) % 100 == 0:
            theta1_deg = np.degrees(obs[1])
            theta2_deg = np.degrees(obs[2])
            print(f"  Step {step+1}/{max_steps}: θ₁={theta1_deg:.1f}°, θ₂={theta2_deg:.1f}°, u={action:.2f}N")
    
    states = np.array(states)
    actions = np.array(actions)
    rewards = np.array(rewards)
    computation_times = np.array(computation_times)
    
    # FIXED: Get actual timestep from environment
    dt = env.dt * env.frame_skip  # = 0.01 * 1 = 0.01 seconds
    
    # Compute metrics
    settling_time = compute_settling_time(states, dt)
    control_effort = np.sum(actions**2)
    final_error = np.abs(states[-1, 1] - np.pi) + np.abs(states[-1, 2] - np.pi)
    
    # NEW: Additional metrics
    peak_force = np.max(np.abs(actions))
    saturation_count = np.sum(np.abs(actions) > 20.0)
    saturation_percent = (saturation_count / len(actions)) * 100 if len(actions) > 0 else 0.0
    
    print(f"\n{'='*60}")
    print(f"Results:")
    print(f"  Steps completed: {len(states)-1}")
    print(f"  Settling time: {settling_time:.2f}s")
    print(f"  Control effort: {control_effort:.2f} N²⋅s")
    print(f"  Peak force: {peak_force:.2f}N")
    print(f"  Saturation: {saturation_count}/{len(actions)} steps ({saturation_percent:.1f}%)")
    print(f"  Final angle error: {np.degrees(final_error):.2f}°")
    print(f"  Avg computation time: {np.mean(computation_times)*1000:.3f}ms")
    print(f"  Max computation time: {np.max(computation_times)*1000:.3f}ms")
    print(f"{'='*60}")
    
    return {
        'name': name,
        'states': states,
        'actions': actions,
        'rewards': rewards,
        'computation_times': computation_times,
        'settling_time': settling_time,
        'control_effort': control_effort,
        'final_error': final_error,
        'initial_state': initial_state,
        'dt': dt,
        'peak_force': peak_force,
        'saturation_count': saturation_count,
        'saturation_percent': saturation_percent,
    }


def compute_settling_time(states, dt, threshold_deg=3.0):
    """
    Compute settling time (when angles stay within threshold).
    
    Args:
        states: State trajectory
        dt: Timestep in seconds
        threshold_deg: Threshold in degrees
    
    Returns:
        Settling time in seconds
    """
    threshold_rad = np.radians(threshold_deg)
    
    for i in range(len(states)):
        theta1_error = abs(states[i, 1] - np.pi)
        theta2_error = abs(states[i, 2] - np.pi)
        
        # Check if remaining trajectory stays within threshold
        if i < len(states) - 20:  # Need at least 20 steps ahead
            future_theta1 = np.abs(states[i:i+20, 1] - np.pi)
            future_theta2 = np.abs(states[i:i+20, 2] - np.pi)
            
            if np.all(future_theta1 < threshold_rad) and np.all(future_theta2 < threshold_rad):
                return i * dt
    
    return len(states) * dt  # Did not settle


def plot_experiment_results(result, save_path=None):
    """
    Create comprehensive plots for a single experiment.
    
    Args:
        result: Dictionary from run_lqr_experiment
        save_path: Path to save figure
    """
    states = result['states']
    actions = result['actions']
    rewards = result['rewards']
    dt = result['dt']  # Get dt from result
    
    # FIXED: Use correct timestep
    time = np.arange(len(states)) * dt
    time_actions = time[:-1]
    
    # Create figure with better spacing
    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.3, 
                          top=0.94, bottom=0.06, left=0.08, right=0.96)
    
    # Suptitle with more space
    fig.suptitle(f'LQR Controller - {result["name"]}', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # 1. Cart position
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(time, states[:, 0], 'b-', linewidth=2)
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax1.set_ylabel('Cart Position (m)', fontsize=10)
    ax1.set_xlabel('Time (s)', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Cart Position', fontweight='bold', fontsize=11, pad=1)
    
    # 2. Angles
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(time, np.degrees(states[:, 1]), 'r-', linewidth=2, label='θ₁')
    ax2.plot(time, np.degrees(states[:, 2]), 'g-', linewidth=2, label='θ₂')
    ax2.axhline(y=180, color='b', linestyle='--', alpha=0.5, linewidth=1.5, label='Upright')
    ax2.fill_between(time, 177, 183, alpha=0.2, color='green', label='±3°')
    ax2.set_ylabel('Angle (degrees)', fontsize=10)
    ax2.set_xlabel('Time (s)', fontsize=10)
    ax2.legend(loc='best', fontsize=8, framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Pendulum Angles', fontweight='bold', fontsize=11, pad=1)
    
    # 3. Velocities - DUAL Y-AXIS
    ax3 = fig.add_subplot(gs[1, 0])

    # Left y-axis: Cart velocity (m/s)
    ax3.plot(time, states[:, 3], 'b-', linewidth=2, label='ẋ (cart)')
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax3.set_ylabel('Cart Velocity (m/s)', fontsize=10, color='b')
    ax3.set_xlabel('Time (s)', fontsize=10)
    ax3.tick_params(axis='y', labelcolor='b')
    ax3.grid(True, alpha=0.3)

    # Right y-axis: Angular velocities (rad/s)
    ax3_right = ax3.twinx()
    ax3_right.plot(time, states[:, 4], 'r-', linewidth=2, label='θ̇₁', alpha=0.8)
    ax3_right.plot(time, states[:, 5], 'g-', linewidth=2, label='θ̇₂', alpha=0.8)
    ax3_right.set_ylabel('Angular Velocity (rad/s)', fontsize=10, color='r')
    ax3_right.tick_params(axis='y', labelcolor='r')

    # Combine legends from both axes
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_right.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper right', 
            fontsize=8, framealpha=0.9)

    ax3.set_title('System Velocities', fontweight='bold', fontsize=11, pad=1)
    
    # 4. Control input
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(time_actions, actions, 'purple', linewidth=2)
    ax4.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax4.axhline(y=20, color='r', linestyle='--', alpha=0.5, linewidth=1, label='±20N limit')
    ax4.axhline(y=-20, color='r', linestyle='--', alpha=0.5, linewidth=1)
    ax4.set_ylabel('Control Force (N)', fontsize=10)
    ax4.set_xlabel('Time (s)', fontsize=10)
    ax4.legend(loc='best', fontsize=8, framealpha=0.9)
    ax4.grid(True, alpha=0.3)
    ax4.set_title('Control Input', fontweight='bold', fontsize=11, pad=1)
    
    # 5. Phase portrait - θ₁
    ax5 = fig.add_subplot(gs[2, 0])
    theta1_deg = np.degrees(states[:, 1])
    dtheta1 = states[:, 4]
    scatter = ax5.scatter(theta1_deg, dtheta1, c=time, cmap='viridis', 
                         s=15, alpha=0.5, edgecolors='none')
    ax5.plot(180, 0, 'r*', markersize=12, label='Equilibrium', zorder=10)
    ax5.set_xlabel('θ₁ (degrees)', fontsize=10)
    ax5.set_ylabel('θ̇₁ (rad/s)', fontsize=10)
    ax5.legend(loc='best', fontsize=8, framealpha=0.9)
    ax5.grid(True, alpha=0.3)
    ax5.set_title('Phase Portrait - Link 1', fontweight='bold', fontsize=11, pad=1)
    cbar1 = plt.colorbar(scatter, ax=ax5)
    cbar1.set_label('Time (s)', fontsize=9)
    
    # 6. Phase portrait - θ₂
    ax6 = fig.add_subplot(gs[2, 1])
    theta2_deg = np.degrees(states[:, 2])
    dtheta2 = states[:, 5]
    scatter = ax6.scatter(theta2_deg, dtheta2, c=time, cmap='plasma', 
                         s=15, alpha=0.5, edgecolors='none')
    ax6.plot(180, 0, 'r*', markersize=12, label='Equilibrium', zorder=10)
    ax6.set_xlabel('θ₂ (degrees)', fontsize=10)
    ax6.set_ylabel('θ̇₂ (rad/s)', fontsize=10)
    ax6.legend(loc='best', fontsize=8, framealpha=0.9)
    ax6.grid(True, alpha=0.3)
    ax6.set_title('Phase Portrait - Link 2', fontweight='bold', fontsize=11, pad=1)
    cbar2 = plt.colorbar(scatter, ax=ax6)
    cbar2.set_label('Time (s)', fontsize=9)
    
    # 7. Metrics summary - SMALLER BOX
    ax7 = fig.add_subplot(gs[3, :])
    ax7.axis('off')
    
    metrics_text = f"""Performance Metrics - {result['name']}

Initial: x={result['initial_state'][0]:.3f}m, θ₁={np.degrees(result['initial_state'][1]):.1f}°, θ₂={np.degrees(result['initial_state'][2]):.1f}°
Settling: {result['settling_time']:.2f}s | Control Effort: {result['control_effort']:.2f} N²⋅s | Peak Force: {result['peak_force']:.2f}N | Saturation: {result['saturation_percent']:.1f}%
Final Error: {np.degrees(result['final_error']):.2f}° | Computation: {np.mean(result['computation_times'])*1000:.3f}ms avg, {np.max(result['computation_times'])*1000:.3f}ms max | Steps: {len(states)-1}"""
    
    ax7.text(0.5, 0.5, metrics_text, fontsize=10, verticalalignment='center',
            horizontalalignment='center', fontfamily='monospace', 
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3, pad=1.75))
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"✓ Plot saved to {save_path}")
    
    return fig


def create_animation(result, save_path=None):
    """
    Create animation of the LQR-controlled pendulum.
    
    Args:
        result: Dictionary from run_lqr_experiment
        save_path: Path to save animation
    """
    states = result['states']
    actions = result['actions']
    dt = result['dt']  # Get dt from result (0.01s)
    
    print(f"\nCreating animation with {len(states)} frames...")
    
    # Create figure with larger size and better styling
    fig, (ax_main, ax_ctrl) = plt.subplots(1, 2, figsize=(14, 8))
    
    # Main animation subplot
    ax_main.set_xlim(-2.2, 2.2)
    ax_main.set_ylim(-1.0, 1.5)
    ax_main.set_aspect('equal')
    ax_main.grid(True, alpha=0.3)
    ax_main.set_xlabel('x (m)', fontsize=12)
    ax_main.set_ylabel('y (m)', fontsize=12)
    ax_main.set_title(f'LQR Control - {result["name"]}', fontsize=14, fontweight='bold')
    
    # Draw rail
    ax_main.plot([-2, 2], [0, 0], 'k-', linewidth=2, alpha=0.3, label='Rail')
    
    # Cart with better styling
    cart_width, cart_height = 0.4, 0.15
    cart_patch = plt.Rectangle((0, 0), cart_width, cart_height, 
                               fc='steelblue', ec='black', linewidth=2, alpha=0.8)
    ax_main.add_patch(cart_patch)
    
    # First rod and blob: RED with darker face
    line1, = ax_main.plot([], [], 'o-', color='red', linewidth=4, markersize=10, 
                          markerfacecolor='darkred', label='Link 1')
    
    # Second rod and blob: BLUE with darker face
    line2, = ax_main.plot([], [], 'o-', color='blue', linewidth=4, markersize=10,
                          markerfacecolor='darkblue', label='Link 2')
    
    # Remove yellow tip marker (invisible)
    tip_marker, = ax_main.plot([], [], 'o', color='none', markersize=0)
    
    # Trajectory traces: red for link 1, green for link 2
    trace1, = ax_main.plot([], [], 'r-', alpha=0.3, linewidth=1.5)
    trace2, = ax_main.plot([], [], 'g-', alpha=0.3, linewidth=1.5)
    
    info_text = ax_main.text(0.02, 0.98, '', transform=ax_main.transAxes,
                            verticalalignment='top', fontsize=10,
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
    ax_main.legend(loc='upper right', fontsize=9)
    
    # Control subplot
    ax_ctrl.set_xlim(0, len(states) * dt)
    ax_ctrl.set_ylim(-25, 25)
    ax_ctrl.set_xlabel('Time (s)', fontsize=11)
    ax_ctrl.set_ylabel('Control Force (N)', fontsize=11)
    ax_ctrl.set_title('Control Input History', fontsize=12, fontweight='bold')
    ax_ctrl.grid(True, alpha=0.3)
    ax_ctrl.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax_ctrl.axhline(y=20, color='r', linestyle='--', alpha=0.3)
    ax_ctrl.axhline(y=-20, color='r', linestyle='--', alpha=0.3)
    
    ctrl_line, = ax_ctrl.plot([], [], 'purple', linewidth=2)
    ctrl_marker, = ax_ctrl.plot([], [], 'ro', markersize=8)
    
    # Traces for visualization
    trace1_x, trace1_y = [], []
    trace2_x, trace2_y = [], []
    
    def init():
        cart_patch.set_xy((0, 0))
        line1.set_data([], [])
        line2.set_data([], [])
        tip_marker.set_data([], [])
        trace1.set_data([], [])
        trace2.set_data([], [])
        ctrl_line.set_data([], [])
        ctrl_marker.set_data([], [])
        info_text.set_text('')
        return (cart_patch, line1, line2, tip_marker, trace1, trace2,
                ctrl_line, ctrl_marker, info_text)
    
    def animate(frame):
        if frame >= len(states):
            frame = len(states) - 1
            
        x, theta1, theta2 = states[frame, :3]
        
        # Update cart
        cart_patch.set_xy((x - cart_width/2, -cart_height/2))
        
        # First pendulum
        x1 = x + 0.5 * np.sin(theta1)
        y1 = -0.5 * np.cos(theta1)
        line1.set_data([x, x1], [0, y1])
        
        # Second pendulum
        x2 = x + 0.5 * np.sin(theta1) + 0.4 * np.sin(theta2)
        y2 = -0.5 * np.cos(theta1) - 0.4 * np.cos(theta2)
        line2.set_data([x1, x2], [y1, y2])
        tip_marker.set_data([x2], [y2])
        
        # Update traces
        trace1_x.append(x1)
        trace1_y.append(y1)
        trace2_x.append(x2)
        trace2_y.append(y2)
        
        # Keep only last 50 points
        if len(trace1_x) > 50:
            trace1_x.pop(0)
            trace1_y.pop(0)
            trace2_x.pop(0)
            trace2_y.pop(0)
        
        trace1.set_data(trace1_x, trace1_y)
        trace2.set_data(trace2_x, trace2_y)
        
        # Update control plot
        if frame > 0 and len(actions) > 0:
            n_actions = min(frame, len(actions))
            time_ctrl = np.arange(n_actions) * dt
            ctrl_line.set_data(time_ctrl, actions[:n_actions])
            if n_actions > 0:
                ctrl_marker.set_data([time_ctrl[-1]], [actions[n_actions-1]])
        
        # Info text
        t = frame * dt
        u = actions[frame-1] if frame > 0 and frame-1 < len(actions) else 0.0
        
        info_text.set_text(
            f'Time: {t:.2f}s | Frame: {frame}/{len(states)-1}\n'
            f'Cart: x={x:.3f}m\n'
            f'θ₁={np.degrees(theta1):.1f}° | θ₂={np.degrees(theta2):.1f}°\n'
            f'Control: {u:.2f}N'
        )
        
        return (cart_patch, line1, line2, tip_marker, trace1, trace2,
                ctrl_line, ctrl_marker, info_text)
    
    anim = FuncAnimation(fig, animate, init_func=init, frames=len(states),
                        interval=50, blit=True)
    
    if save_path:
        print(f"Saving animation to {save_path}...")
        anim.save(save_path, writer='pillow', fps=20)
        print(f"✓ Animation saved!")
    
    return fig, anim


def main():
    """Run comprehensive LQR experiments."""
    print("\n" + "🎯 " * 20)
    print("LQR CONTROLLER EXPERIMENTS")
    print("🎯 " * 20)
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results/lqr_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    print(f"\nResults will be saved to: {results_dir}")
    
    # Create controller
    print("\n" + "="*60)
    print("Creating LQR Controller...")
    print("="*60)
    controller = create_lqr_controller()
    
    # Create environment - NO termination to see full LQR performance
    env = DoublePendulumCartEnv(terminate_on_fall=False)
    
    # Define test scenarios - start closer to upright for LQR
    scenarios = [
        {
            'name': 'Small Perturbation',
            'initial_state': np.array([0.0, np.pi + 0.05, np.pi + 0.03, 0.0, 0.0, 0.0]),
            'max_steps': 400
        },
        {
            'name': 'Medium Perturbation',
            'initial_state': np.array([0.1, np.pi + 0.15, np.pi + 0.1, 0.0, 0.0, 0.0]),
            'max_steps': 500
        },
        {
            'name': 'Large Perturbation',
            'initial_state': np.array([0.0, np.pi + 0.25, np.pi + 0.2, 0.0, 0.1, 0.0]),
            'max_steps': 600
        },
    ]
    
    # Run experiments
    results = []
    for scenario in scenarios:
        result = run_lqr_experiment(
            controller, env, 
            scenario['initial_state'],
            scenario['max_steps'],
            scenario['name']
        )
        results.append(result)
        
        # Create plots
        plot_path = os.path.join(results_dir, f"{scenario['name'].replace(' ', '_').lower()}_plot.png")
        plot_experiment_results(result, plot_path)
        plt.close()
        
        # Create animation
        anim_path = os.path.join(results_dir, f"{scenario['name'].replace(' ', '_').lower()}_animation.gif")
        fig, anim = create_animation(result, anim_path)
        plt.close(fig)
        
        controller.reset()  # Reset controller history
    
    
    print("\n" + "✅ " * 20)
    print("ALL EXPERIMENTS COMPLETE!")
    print("✅ " * 20)
    print(f"\nResults saved to: {results_dir}")
    print("\nGenerated files:")
    for scenario in scenarios:
        name = scenario['name'].replace(' ', '_').lower()
        print(f"  • {name}_plot.png")
        print(f"  • {name}_animation.gif")
    


if __name__ == "__main__":
    main()
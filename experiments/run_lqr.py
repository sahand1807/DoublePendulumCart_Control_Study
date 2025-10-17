"""
LQR Controller Experiment Runner with MuJoCo 3D Visualization

Tests the LQR controller on the double pendulum cart system with various
initial conditions and produces comprehensive visualizations, including
real-time 3D rendering in MuJoCo.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sys
import os
from datetime import datetime
import mujoco
import mujoco_viewer

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.double_pendulum_cart_env import DoublePendulumCartEnv
from controllers.lqr_controller import create_lqr_controller


def run_lqr_experiment(controller, env, initial_state, max_steps=500, name="test", render_3d=False):
    """
    Run a single LQR experiment with optional MuJoCo 3D rendering.

    Args:
        controller: LQR controller instance
        env: Gym environment
        initial_state: Initial state [x, θ₁, θ₂, ẋ, θ̇₁, θ̇₂]
        max_steps: Maximum number of steps
        name: Experiment name for logging
        render_3d: Whether to render in MuJoCo viewer

    Returns:
        Dictionary with trajectory data
    """
    print(f"\n{'='*60}")
    print(f"Running LQR Experiment: {name}")
    print(f"{'='*60}")
    print(f"Initial state: x={initial_state[0]:.3f}, θ₁={np.degrees(initial_state[1]):.1f}°, θ₂={np.degrees(initial_state[2]):.1f}°")
    
    # Initialize MuJoCo viewer if rendering
    viewer = None
    if render_3d:
        print("\nInitializing MuJoCo 3D viewer...")
        print("🖱️ Controls: Left drag (rotate), Right drag (pan), Scroll (zoom), ESC (close)")
        viewer = mujoco_viewer.MujocoViewer(env.model, env.data)
    
    # Reset environment
    obs, _ = env.reset(options={"initial_state": initial_state})
    
    # Storage
    states = [obs.copy()]
    actions = []
    rewards = []
    computation_times = []
    
    # Run episode
    for step in range(max_steps):
        # Compute control (raw force in Newtons)
        action, comp_time = controller.compute_control_timed(obs)
        
        # Normalize and clip for environment (env expects [-1, 1], scales by max_force)
        action_clipped = np.clip(action / env.max_force, -1.0, 1.0)
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(np.array([action_clipped]))
        
        # Store data
        states.append(obs.copy())
        actions.append(action)
        rewards.append(reward)
        computation_times.append(comp_time)
        controller.log_step(obs, np.array([action]))
        
        # Render in MuJoCo viewer
        if render_3d and viewer.is_alive:
            viewer.render()
            # Print state occasionally
            if step % 100 == 0:
                t = step * env.dt * env.frame_skip
                x = obs[0]
                theta1_abs = obs[1]
                theta2_abs = obs[2]
                print(f"t={t:5.2f}s: x={x:6.3f}m, "
                      f"θ₁={np.degrees(theta1_abs):7.2f}°, "
                      f"θ₂={np.degrees(theta2_abs):7.2f}°, "
                      f"u={action:.2f}N")
        
        # Check termination
        if terminated or truncated:
            print(f"Episode ended at step {step+1}")
            break
        
        # Progress
        if (step + 1) % 100 == 0:
            theta1_deg = np.degrees(obs[1])
            theta2_deg = np.degrees(obs[2])
            print(f"  Step {step+1}/{max_steps}: θ₁={theta1_deg:.1f}°, θ₂={theta2_deg:.1f}°, u={action:.2f}N")
    
    # Close viewer
    if render_3d:
        viewer.close()
        print("✓ MuJoCo viewer closed")
    
    states = np.array(states)
    actions = np.array(actions)
    rewards = np.array(rewards)
    computation_times = np.array(computation_times)
    
    # Compute metrics
    settling_time = compute_settling_time(states)
    control_effort = np.sum(actions**2)
    final_error = np.abs(states[-1, 1] - np.pi) + np.abs(states[-1, 2] - np.pi)
    
    print(f"\n{'='*60}")
    print(f"Results:")
    print(f"  Steps completed: {len(states)-1}")
    print(f"  Settling time: {settling_time:.2f}s")
    print(f"  Control effort: {control_effort:.2f}")
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
        'initial_state': initial_state
    }


# [Rest of the file unchanged: compute_settling_time, plot_experiment_results, create_animation]

def main():
    """Run comprehensive LQR experiments with MuJoCo 3D rendering."""
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
    
    # Create environment with rendering enabled
    env = DoublePendulumCartEnv(terminate_on_fall=False, render_mode="human")
    
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
            scenario['name'],
            render_3d=True  # Enable MuJoCo rendering
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
    
    # Show one plot
    print("\nDisplaying results...")
    plt.show()


if __name__ == "__main__":
    main()
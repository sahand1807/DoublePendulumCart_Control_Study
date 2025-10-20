"""
Render trained PPO swing-up model and create animated visualization.

Shows the controller swinging up the double pendulum from hanging position (θ=0°)
to inverted position (θ=180°) and then stabilizing.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import mujoco
from stable_baselines3 import PPO
from env.double_pendulum_cart_env import DoublePendulumCartEnv
from env.angle_wrapper import AngleObservationWrapper
from env.swing_up_wrapper import SwingUpInitializationWrapper, SwingUpRewardWrapper
import matplotlib.pyplot as plt
import matplotlib.animation as animation


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


def render_episode(model, env, max_steps=5000):
    """
    Run one swing-up episode and collect frames and metrics.

    Args:
        model: Trained PPO model
        env: Swing-up environment
        max_steps: Maximum episode steps

    Returns:
        frames: List of rendered images
        metrics: Dict with trajectory data
    """
    obs, info = env.reset()

    # Get initial state
    x_0 = obs[0]
    theta1_0 = np.arctan2(obs[1], obs[2])
    theta2_0 = np.arctan2(obs[3], obs[4])

    # Convert to degrees for display (from hanging position)
    theta1_deg_0 = np.degrees(theta1_0)
    theta2_deg_0 = np.degrees(theta2_0)

    print(f"  Initial position: θ₁={theta1_deg_0:+.2f}° (hanging=0°), θ₂={theta2_deg_0:+.2f}°")

    # Storage
    frames = []
    states = []
    actions_list = []
    rewards_list = []

    # Setup renderer
    renderer = mujoco.Renderer(env.unwrapped.model, height=480, width=640)

    # Configure camera for better view of swing-up motion
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(env.unwrapped.model, camera)

    # Camera view - zoom out more to see full swing motion
    camera.lookat[0] = 0.0
    camera.lookat[1] = 0.0
    camera.lookat[2] = 1.0
    camera.distance = 4.5  # Zoom out more for swing-up
    camera.elevation = -10
    camera.azimuth = 90

    renderer.update_scene(env.unwrapped.data, camera=camera)

    done = False
    step = 0
    swing_up_achieved = False
    swing_up_step = None

    while not done and step < max_steps:
        # Get action
        action, _ = model.predict(obs, deterministic=True)

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Extract actual state from observation
        x = obs[0]
        theta1 = np.arctan2(obs[1], obs[2])
        theta2 = np.arctan2(obs[3], obs[4])
        dx, dtheta1, dtheta2 = obs[5], obs[6], obs[7]
        states.append([x, theta1, theta2, dx, dtheta1, dtheta2])

        actions_list.append(action[0])
        rewards_list.append(reward)

        # Check if swing-up achieved (within 10° of upright)
        if not swing_up_achieved:
            theta1_error = abs(theta1 - np.pi)
            theta2_error = abs(theta2 - np.pi)
            if theta1_error < np.deg2rad(10) and theta2_error < np.deg2rad(10):
                swing_up_achieved = True
                swing_up_step = step
                print(f"  Swing-up achieved at step {step} ({step*0.005:.2f}s)")

        # Render frame every 5 steps
        if step % 5 == 0:
            renderer.update_scene(env.unwrapped.data, camera=camera)
            pixels = renderer.render()
            frames.append(pixels.copy())

        step += 1

    renderer.close()

    metrics = {
        'states': np.array(states),
        'actions': np.array(actions_list),
        'rewards': np.array(rewards_list),
        'length': step,
        'total_reward': np.sum(rewards_list),
        'swing_up_step': swing_up_step,
        'swing_up_time': swing_up_step * 0.005 if swing_up_step else None
    }

    print(f"  Episode length: {step} steps ({step*0.005:.1f}s)")
    print(f"  Total reward: {metrics['total_reward']:.1f}")
    if swing_up_step:
        print(f"  Swing-up time: {metrics['swing_up_time']:.2f}s")

    return frames, metrics


def create_animation(frames, metrics, save_path):
    """Create animated visualization with frames and plots."""

    fig = plt.figure(figsize=(16, 9))

    # Layout: 2x3 grid
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    ax_video = fig.add_subplot(gs[0, :2])
    ax_angles = fig.add_subplot(gs[0, 2])
    ax_cart = fig.add_subplot(gs[1, 0])
    ax_control = fig.add_subplot(gs[1, 1])
    ax_text = fig.add_subplot(gs[1, 2])
    ax_text.axis('off')

    # Extract metrics
    states = metrics['states']
    actions = metrics['actions']

    time = np.arange(len(states)) * 0.005  # 200 Hz

    # Compute angles (convert to degrees, 0° = hanging, 180° = upright)
    theta1 = states[:, 1]
    theta2 = states[:, 2]
    theta1_deg = np.degrees(theta1)
    theta2_deg = np.degrees(theta2)

    # Setup video display
    im = ax_video.imshow(frames[0])
    ax_video.set_title('PPO Swing-Up Controller - Real-time Visualization', fontsize=14, fontweight='bold')
    ax_video.axis('off')

    # Angles plot (absolute angles, not errors)
    line_theta1, = ax_angles.plot([], [], 'b-', label='θ₁', linewidth=2)
    line_theta2, = ax_angles.plot([], [], 'r-', label='θ₂', linewidth=2)
    ax_angles.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1, label='Hanging')
    ax_angles.axhline(y=180, color='green', linestyle='--', alpha=0.5, linewidth=1, label='Upright')
    if metrics['swing_up_step']:
        ax_angles.axvline(x=metrics['swing_up_time'], color='orange', linestyle=':',
                         alpha=0.7, linewidth=2, label='Swing-up')
    ax_angles.set_xlim(0, time[-1])
    ax_angles.set_ylim(-10, 190)
    ax_angles.set_xlabel('Time (s)', fontsize=10)
    ax_angles.set_ylabel('Angle (deg)', fontsize=10)
    ax_angles.set_title('Pendulum Angles', fontsize=12)
    ax_angles.legend(loc='upper right', fontsize=8)
    ax_angles.grid(True, alpha=0.3)

    # Cart position plot
    line_cart, = ax_cart.plot([], [], 'g-', linewidth=2)
    ax_cart.axhline(y=0, color='k', linestyle='-', alpha=0.2, linewidth=1)
    ax_cart.set_xlim(0, time[-1])
    ax_cart.set_ylim(np.min(states[:, 0])-0.1, np.max(states[:, 0])+0.1)
    ax_cart.set_xlabel('Time (s)', fontsize=10)
    ax_cart.set_ylabel('Position (m)', fontsize=10)
    ax_cart.set_title('Cart Position', fontsize=12)
    ax_cart.grid(True, alpha=0.3)

    # Control force plot
    line_control, = ax_control.plot([], [], 'purple', linewidth=2)
    ax_control.axhline(y=0, color='k', linestyle='-', alpha=0.2, linewidth=1)
    ax_control.set_xlim(0, time[-1])
    ax_control.set_ylim(np.min(actions)-1, np.max(actions)+1)
    ax_control.set_xlabel('Time (s)', fontsize=10)
    ax_control.set_ylabel('Force (N)', fontsize=10)
    ax_control.set_title('Control Input', fontsize=12)
    ax_control.grid(True, alpha=0.3)

    # Initial metrics text
    text_content = ax_text.text(0.1, 0.5, '', fontsize=11, verticalalignment='center',
                                 family='monospace')

    def init():
        im.set_data(frames[0])
        line_theta1.set_data([], [])
        line_theta2.set_data([], [])
        line_cart.set_data([], [])
        line_control.set_data([], [])
        return im, line_theta1, line_theta2, line_cart, line_control, text_content

    def animate(i):
        # Update video (every 5 steps)
        frame_idx = min(i, len(frames) - 1)
        im.set_data(frames[frame_idx])

        # Update plots (all steps)
        step_idx = min(i * 5, len(time) - 1)

        line_theta1.set_data(time[:step_idx], theta1_deg[:step_idx])
        line_theta2.set_data(time[:step_idx], theta2_deg[:step_idx])
        line_cart.set_data(time[:step_idx], states[:step_idx, 0])
        line_control.set_data(time[:step_idx], actions[:step_idx])

        # Update metrics text
        current_time = time[step_idx]
        current_theta1 = theta1_deg[step_idx]
        current_theta2 = theta2_deg[step_idx]

        # Determine phase
        if metrics['swing_up_step'] and step_idx < metrics['swing_up_step']:
            phase = "SWINGING UP"
        elif metrics['swing_up_step']:
            phase = "STABILIZING"
        else:
            phase = "IN PROGRESS"

        text_str = f"""
SWING-UP TASK

Phase: {phase}
Time: {current_time:.2f}s
Steps: {step_idx}

Current Angles:
  θ₁: {current_theta1:.1f}°
  θ₂: {current_theta2:.1f}°
  (0°=down, 180°=up)

"""
        if metrics['swing_up_step']:
            text_str += f"Swing-up: {metrics['swing_up_time']:.2f}s\n"

        text_str += f"\nTotal Reward: {np.sum(metrics['rewards'][:step_idx]):.0f}"

        text_content.set_text(text_str)

        return im, line_theta1, line_theta2, line_cart, line_control, text_content

    # Create animation
    n_frames = len(frames)
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=n_frames, interval=50, blit=True)

    # Save
    print(f"\nSaving animation to {save_path}...")
    anim.save(save_path, writer='pillow', fps=20, dpi=100)
    print(f"Animation saved!")

    plt.close()


def main():
    # Get project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # Configuration
    MODEL_PATH = 'results/ppo_swingup/best_model/best_model'
    OUTPUT_DIR = 'results/ppo_swingup'
    OUTPUT_FILENAME = 'swingup_animation.gif'
    MAX_STEPS = 5000  # Longer episodes for swing-up
    MAX_EPISODE_STEPS = 10000

    # Build full paths
    model_path = os.path.join(project_root, MODEL_PATH)
    save_dir = os.path.join(project_root, OUTPUT_DIR)
    save_path = os.path.join(save_dir, OUTPUT_FILENAME)

    print("\n" + "="*70)
    print("PPO Swing-Up Controller - Rendering Visualization")
    print("="*70)
    print("\nTask: Swing up double pendulum from hanging (0°) to upright (180°)")
    print("Control: Horizontal force on cart")
    print("="*70)

    # Load model
    print(f"\nLoading model: {model_path}")
    model = PPO.load(model_path)

    # Create environment
    print(f"Creating swing-up environment...")
    env = create_swingup_env(max_episode_steps=MAX_EPISODE_STEPS)

    # Render episode
    print(f"\nRendering episode (max {MAX_STEPS} steps = {MAX_STEPS*0.005:.1f}s)...")
    frames, metrics = render_episode(model, env, max_steps=MAX_STEPS)

    env.close()

    # Create animation
    create_animation(frames, metrics, save_path)

    print("\n" + "="*70)
    print("Visualization complete!")
    print(f"Animation saved to: {save_path}")
    print("="*70)


if __name__ == '__main__':
    main()

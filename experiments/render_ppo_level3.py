"""
Render trained PPO Level 3 model and create animated visualization.
Shows the controller stabilizing the double pendulum from ±30° perturbations.
"""
# Add parent directory to path for imports when running from experiments/
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import mujoco
from stable_baselines3 import PPO
from env.double_pendulum_cart_env import DoublePendulumCartEnv
from env.angle_wrapper import AngleObservationWrapper, CurriculumInitializationWrapper
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def create_level3_env():
    """Create Level 3 environment with ±30° perturbations."""
    env = DoublePendulumCartEnv()
    env = AngleObservationWrapper(env)
    env = CurriculumInitializationWrapper(env, curriculum_level=3)
    return env

def render_episode(model, env, max_steps=1000):
    """
    Run one episode and collect frames and metrics.

    Returns:
        frames: List of rendered images
        metrics: Dict with trajectory data
    """
    obs, info = env.reset()

    # Get initial state
    if 'actual_state' in info:
        _, theta1_0, theta2_0, _, _, _ = info['actual_state']
    else:
        theta1_0 = np.arctan2(obs[1], obs[2])
        theta2_0 = np.arctan2(obs[3], obs[4])

    # Convert to degrees for display
    theta1_error_0 = np.degrees(np.arctan2(np.sin(theta1_0 - np.pi), np.cos(theta1_0 - np.pi)))
    theta2_error_0 = np.degrees(np.arctan2(np.sin(theta2_0 - np.pi), np.cos(theta2_0 - np.pi)))

    print(f"  Initial perturbations: θ₁={theta1_error_0:+.2f}°, θ₂={theta2_error_0:+.2f}°")

    # Storage
    frames = []
    states = []
    actions_list = []
    rewards_list = []

    # Setup renderer
    renderer = mujoco.Renderer(env.unwrapped.model, height=480, width=640)

    # Configure camera for zoomed-in view
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(env.unwrapped.model, camera)

    # Zoom in on the cart and pendulum
    camera.lookat[0] = 0.0   # x: center on cart
    camera.lookat[1] = 0.0   # y: center
    camera.lookat[2] = 1.0   # z: look at middle height
    camera.distance = 2.5    # Zoom in (reduced from default ~4-5)
    camera.elevation = -10   # Slightly below horizontal
    camera.azimuth = 90      # Side view

    # Update scene with camera
    renderer.update_scene(env.unwrapped.data, camera=camera)

    done = False
    step = 0

    while not done and step < max_steps:
        # Get action
        action, _ = model.predict(obs, deterministic=True)

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Store metrics
        if 'actual_state' in info:
            states.append(info['actual_state'])
        else:
            x = obs[0]
            theta1 = np.arctan2(obs[1], obs[2])
            theta2 = np.arctan2(obs[3], obs[4])
            dx, dtheta1, dtheta2 = obs[5], obs[6], obs[7]
            states.append([x, theta1, theta2, dx, dtheta1, dtheta2])

        actions_list.append(action[0])
        rewards_list.append(reward)

        # Render frame every 5 steps (for smoother playback)
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
        'total_reward': np.sum(rewards_list)
    }

    print(f"  Episode length: {step} steps, Total reward: {metrics['total_reward']:.1f}")

    return frames, metrics

def create_animation(frames, metrics, save_path):
    """Create animated visualization with frames and plots."""

    fig = plt.figure(figsize=(16, 9))

    # Layout: 2x3 grid
    # Top row: video (spans 2 columns) + angle errors
    # Bottom row: cart position + control force + metrics text

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

    time = np.arange(len(states)) * 0.01  # 100 Hz

    # Compute angle errors
    theta1 = states[:, 1]
    theta2 = states[:, 2]
    theta1_error = np.degrees(np.arctan2(np.sin(theta1 - np.pi), np.cos(theta1 - np.pi)))
    theta2_error = np.degrees(np.arctan2(np.sin(theta2 - np.pi), np.cos(theta2 - np.pi)))

    # Setup video display
    im = ax_video.imshow(frames[0])
    ax_video.set_title('PPO Level 3 Controller - Real-time Visualization', fontsize=14, fontweight='bold')
    ax_video.axis('off')

    # Angle errors plot
    line_theta1, = ax_angles.plot([], [], 'b-', label='θ₁ error', linewidth=2)
    line_theta2, = ax_angles.plot([], [], 'r-', label='θ₂ error', linewidth=2)
    ax_angles.axhline(y=3, color='k', linestyle='--', alpha=0.3, linewidth=1)
    ax_angles.axhline(y=-3, color='k', linestyle='--', alpha=0.3, linewidth=1)
    ax_angles.axhline(y=0, color='k', linestyle='-', alpha=0.2, linewidth=1)
    ax_angles.set_xlim(0, time[-1])
    ax_angles.set_ylim(-10, 10)
    ax_angles.set_xlabel('Time (s)', fontsize=10)
    ax_angles.set_ylabel('Angle Error (deg)', fontsize=10)
    ax_angles.set_title('Angle Stabilization', fontsize=12)
    ax_angles.legend(loc='upper right')
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

        line_theta1.set_data(time[:step_idx], theta1_error[:step_idx])
        line_theta2.set_data(time[:step_idx], theta2_error[:step_idx])
        line_cart.set_data(time[:step_idx], states[:step_idx, 0])
        line_control.set_data(time[:step_idx], actions[:step_idx])

        # Update metrics text
        current_time = time[step_idx]
        current_theta1_err = theta1_error[step_idx]
        current_theta2_err = theta2_error[step_idx]
        rms_error = np.sqrt(current_theta1_err**2 + current_theta2_err**2)

        text_str = f"""
LEVEL 1: ±30° Perturbations

Time: {current_time:.2f}s
Steps: {step_idx}

Current Errors:
  θ₁: {current_theta1_err:+.2f}°
  θ₂: {current_theta2_err:+.2f}°
  RMS: {rms_error:.2f}°

Total Reward: {metrics['total_reward']:.0f}
        """
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
    # Add parent directory to path for imports when running from experiments/
    import os
    import sys

    # Get project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    sys.path.insert(0, project_root)

    # Model path from training (absolute path from project root)
    # Note: PPO.load() automatically adds .zip extension
    model_path = os.path.join(project_root, 'results/ppo_level3/ppo_level3_final')
    save_dir = os.path.join(project_root, 'results/ppo_level3')

    print("\n" + "="*60)
    print("PPO Level 3 Controller - Rendering Visualization")
    print("="*60)
    print("\nTask: Stabilize double inverted pendulum on cart")
    print("Level 3: ±30° initial perturbations")
    print("Control: Horizontal force on cart")
    print("="*60)

    # Load model
    print(f"\nLoading model: {model_path}")
    model = PPO.load(model_path)

    # Create environment
    env = create_level3_env()

    # Render episode (10 seconds at 100 Hz = 1000 steps)
    print("\nRendering episode...")
    frames, metrics = render_episode(model, env, max_steps=1000)

    env.close()

    # Create animation
    save_path = f"{save_dir}/level3_animation.gif"
    create_animation(frames, metrics, save_path)

    print("\n" + "="*60)
    print("Visualization complete!")
    print(f"Animation saved to: {save_path}")
    print("="*60)

if __name__ == '__main__':
    main()

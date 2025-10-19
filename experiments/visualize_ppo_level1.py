"""
Visualize trained PPO Level 1 model in 3D MuJoCo.
Shows the controller stabilizing the double pendulum from ±3° perturbations.

Options:
  - Interactive viewer (requires mjpython on macOS)
  - Video recording (works with regular python)
"""
import numpy as np
import mujoco
from stable_baselines3 import PPO
from env.double_pendulum_cart_env import DoublePendulumCartEnv
from env.angle_wrapper import AngleObservationWrapper, CurriculumInitializationWrapper
import time
import sys
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import matplotlib.patches as patches

def create_level1_env():
    """Create Level 1 environment with ±3° perturbations."""
    env = DoublePendulumCartEnv()
    env = AngleObservationWrapper(env)
    env = CurriculumInitializationWrapper(env, curriculum_level=1)
    return env

def visualize_controller(model_path, n_episodes=3):
    """
    Visualize the trained controller in MuJoCo viewer.

    Args:
        model_path: Path to trained PPO model
        n_episodes: Number of episodes to run
    """
    print(f"\n{'='*60}")
    print(f"Loading model: {model_path}")
    print(f"{'='*60}\n")

    # Load trained model
    model = PPO.load(model_path)

    # Create environment
    env = create_level1_env()

    # Access the MuJoCo model and data
    mj_model = env.unwrapped.model
    mj_data = env.unwrapped.data

    print("Controls:")
    print("  - Press ESC to close viewer")
    print("  - Click and drag to rotate view")
    print("  - Scroll to zoom")
    print("  - Right-click drag to pan")
    print(f"\nRunning {n_episodes} episodes with Level 1 controller (±3° perturbations)")
    print("="*60)

    # Launch passive viewer
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        # Set camera for better view
        viewer.cam.azimuth = 90
        viewer.cam.distance = 5.0
        viewer.cam.elevation = -20
        viewer.cam.lookat[:] = [0, 0, 1.5]

        for episode in range(n_episodes):
            obs, info = env.reset()

            # Get initial state
            if 'actual_state' in info:
                x0, theta1_0, theta2_0, _, _, _ = info['actual_state']
            else:
                x0 = obs[0]
                theta1_0 = np.arctan2(obs[1], obs[2])
                theta2_0 = np.arctan2(obs[3], obs[4])

            # Convert to degrees for display
            theta1_error_0 = np.degrees(np.arctan2(np.sin(theta1_0 - np.pi), np.cos(theta1_0 - np.pi)))
            theta2_error_0 = np.degrees(np.arctan2(np.sin(theta2_0 - np.pi), np.cos(theta2_0 - np.pi)))

            print(f"\nEpisode {episode+1}/{n_episodes}:")
            print(f"  Initial perturbations: θ₁={theta1_error_0:+.2f}°, θ₂={theta2_error_0:+.2f}°")

            episode_reward = 0
            episode_length = 0
            done = False

            # Run episode
            while not done and viewer.is_running():
                # Get action from trained model
                action, _ = model.predict(obs, deterministic=True)

                # Step environment
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                episode_reward += reward
                episode_length += 1

                # Sync viewer with MuJoCo data
                viewer.sync()

                # Control playback speed (real-time: 0.01s per step)
                time.sleep(0.01)

                # Check if viewer was closed
                if not viewer.is_running():
                    print("\nViewer closed by user.")
                    env.close()
                    return

            print(f"  Episode length: {episode_length} steps")
            print(f"  Total reward: {episode_reward:.1f}")

            # Brief pause between episodes
            if episode < n_episodes - 1 and viewer.is_running():
                print("  Next episode in 2 seconds...")
                for _ in range(200):  # 2 seconds at 100 Hz
                    viewer.sync()
                    time.sleep(0.01)

    env.close()
    print("\n" + "="*60)
    print("Visualization complete!")
    print("="*60)

if __name__ == '__main__':
    model_path = 'results/ppo_fresh/20251019_135042/Level_1_Easy_3deg/Level_1_Easy_3deg_final.zip'

    print("\n" + "="*60)
    print("PPO Level 1 Controller - 3D Visualization")
    print("="*60)
    print("\nTask: Stabilize double inverted pendulum on cart")
    print("Level 1: Random initial perturbations of ±3° around upright")
    print("Control: Horizontal force on cart (-10N to +10N)")
    print("Objective: Keep both pendulums upright (θ₁=π, θ₂=π)")

    visualize_controller(model_path, n_episodes=3)

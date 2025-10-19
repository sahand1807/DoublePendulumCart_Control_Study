"""
PPO Controller with MuJoCo 3D Viewer

Visualizes the trained PPO controller stabilizing the double pendulum in real-time.
Load a trained model and watch it control the system!

Usage:
    python experiments/run_ppo_3D.py --model path/to/model.zip --level 1
"""
import mujoco
import mujoco_viewer
import numpy as np
import sys
import os
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from controllers.ppo_controller import PPOController


def mujoco_to_absolute_angles(qpos):
    """Convert MuJoCo angles (relative) to absolute angles."""
    x = qpos[0]
    theta1_abs = qpos[1]
    theta2_rel = qpos[2]
    theta2_abs = theta1_abs + theta2_rel
    return x, theta1_abs, theta2_abs


def mujoco_to_absolute_velocities(qvel):
    """Convert MuJoCo angular velocities to absolute convention."""
    dx = qvel[0]
    dtheta1 = qvel[1]
    dtheta2_rel = qvel[2]
    dtheta2_abs = dtheta1 + dtheta2_rel
    return dx, dtheta1, dtheta2_abs


def get_state_from_mujoco(data):
    """Extract state in absolute convention [x, θ₁, θ₂, ẋ, θ̇₁, θ̇₂]."""
    x, theta1, theta2 = mujoco_to_absolute_angles(data.qpos)
    dx, dtheta1, dtheta2 = mujoco_to_absolute_velocities(data.qvel)
    return np.array([x, theta1, theta2, dx, dtheta1, dtheta2])


def run_ppo_viewer(model_path, initial_angles_deg=(15.0, -3.5), duration=10.0):
    """
    Run PPO controller visualization.

    Args:
        model_path: Path to trained PPO model (.zip file)
        initial_angles_deg: Tuple of (theta1_deg, theta2_deg) perturbations from upright
        duration: Simulation duration in seconds
    """
    print("=" * 80)
    print("PPO CONTROLLER - 3D MUJOCO VIEWER")
    print("=" * 80)

    # Load MuJoCo model
    print("\nLoading MuJoCo model...")
    model = mujoco.MjModel.from_xml_path("env/double_pendulum_cart.xml")
    data = mujoco.MjData(model)
    print("  ✅ MuJoCo model loaded")

    # Load PPO controller
    print(f"\nLoading PPO controller...")
    print(f"  Model: {model_path}")
    if not os.path.exists(model_path):
        print(f"  ❌ Model not found: {model_path}")
        print(f"\n  Please train a model first using:")
        print(f"     python experiments/train_ppo_curriculum.py")
        return

    controller = PPOController(name="PPO_Viewer", model_path=model_path)
    print("  ✅ PPO controller loaded")

    # Get max force from model
    actuator_id = 0
    max_force = model.actuator_ctrlrange[actuator_id, 1]
    print(f"\n  Max actuator force: ±{max_force}N")

    # Reset simulation
    mujoco.mj_resetData(model, data)

    # Set initial state - perturbation from upright (π rad)
    theta1_init_deg, theta2_init_deg = initial_angles_deg
    theta1_abs = np.pi + np.deg2rad(theta1_init_deg)
    theta2_abs = np.pi + np.deg2rad(theta2_init_deg)

    # Convert to MuJoCo convention
    data.qpos[0] = 0.0  # cart at center
    data.qpos[1] = theta1_abs
    data.qpos[2] = theta2_abs - theta1_abs  # relative angle
    data.qvel[:] = 0.0

    # Forward kinematics
    mujoco.mj_forward(model, data)

    print(f"\n📍 Initial state:")
    print(f"  θ₁ = {np.degrees(theta1_abs):.2f}° (upright = 180.0°)")
    print(f"  θ₂ = {np.degrees(theta2_abs):.2f}° (upright = 180.0°)")
    print(f"  θ₁ error = {theta1_init_deg:+.2f}°")
    print(f"  θ₂ error = {theta2_init_deg:+.2f}°")
    print(f"\n🚀 Starting PPO control...")
    print(f"  Duration: {duration}s")
    print(f"  Control frequency: {1.0/model.opt.timestep:.0f} Hz")
    print(f"\n{'='*80}\n")

    # Create viewer
    viewer = mujoco_viewer.MujocoViewer(model, data, hide_menus=True)

    # Tracking variables
    step = 0
    dt = model.opt.timestep
    sim_time = 0.0
    forces = []
    angle_errors = []
    settling_counter = 0
    settling_threshold = np.deg2rad(3)  # 3 degrees
    settling_time = None

    # Control loop
    while sim_time < duration and viewer.is_alive:
        # Get current state
        state = get_state_from_mujoco(data)
        x, theta1, theta2, dx, dtheta1, dtheta2 = state

        # Compute control action using PPO
        action = controller.compute_control(state)

        # Scale to actual force
        force = action[0] * max_force

        # Apply force (clip to limits)
        data.ctrl[0] = np.clip(force, -max_force, max_force)

        # Track metrics
        forces.append(data.ctrl[0])
        theta1_error = abs(theta1 - np.pi)
        theta2_error = abs(theta2 - np.pi)

        # Wrap errors to [0, π]
        theta1_error = min(theta1_error, 2*np.pi - theta1_error)
        theta2_error = min(theta2_error, 2*np.pi - theta2_error)

        angle_errors.append((theta1_error, theta2_error))

        # Check settling (both angles < 3° for 1 second)
        if theta1_error < settling_threshold and theta2_error < settling_threshold:
            settling_counter += 1
            if settling_counter >= int(1.0 / dt) and settling_time is None:
                settling_time = sim_time
                print(f"✅ SETTLED at t = {settling_time:.2f}s")
                print(f"   θ₁ error: {np.degrees(theta1_error):.3f}°")
                print(f"   θ₂ error: {np.degrees(theta2_error):.3f}°")
        else:
            settling_counter = 0

        # Print periodic status
        if step % 100 == 0:
            print(f"t={sim_time:.2f}s  |  "
                  f"θ₁_err={np.degrees(theta1_error):+6.2f}°  "
                  f"θ₂_err={np.degrees(theta2_error):+6.2f}°  |  "
                  f"u={data.ctrl[0]:+6.2f}N  |  "
                  f"x={x:+5.2f}m")

        # Step simulation
        mujoco.mj_step(model, data)
        viewer.render()

        step += 1
        sim_time += dt

    # Final statistics
    print(f"\n{'='*80}")
    print("SIMULATION COMPLETE")
    print("=" * 80)
    print(f"\nFinal state:")
    state = get_state_from_mujoco(data)
    x, theta1, theta2, dx, dtheta1, dtheta2 = state
    theta1_error = abs(theta1 - np.pi)
    theta2_error = abs(theta2 - np.pi)
    theta1_error = min(theta1_error, 2*np.pi - theta1_error)
    theta2_error = min(theta2_error, 2*np.pi - theta2_error)

    print(f"  θ₁ = {np.degrees(theta1):.2f}° (error: {np.degrees(theta1_error):.3f}°)")
    print(f"  θ₂ = {np.degrees(theta2):.2f}° (error: {np.degrees(theta2_error):.3f}°)")
    print(f"  Cart position: {x:.3f} m")

    print(f"\nControl statistics:")
    forces = np.array(forces)
    print(f"  Mean |force|: {np.mean(np.abs(forces)):.2f} N")
    print(f"  Max |force|: {np.max(np.abs(forces)):.2f} N")
    print(f"  Force std: {np.std(forces):.2f} N")

    saturated_steps = np.sum(np.abs(forces) >= max_force * 0.99)
    saturation_pct = saturated_steps / len(forces) * 100
    print(f"  Saturation: {saturated_steps}/{len(forces)} steps ({saturation_pct:.1f}%)")

    if settling_time:
        print(f"\nSettling time: {settling_time:.2f}s")
    else:
        print(f"\n⚠️  System did not settle within {duration}s")

    print("\n" + "=" * 80)

    # Close viewer
    viewer.close()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize PPO controller in 3D MuJoCo viewer"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained PPO model (.zip file)",
    )
    parser.add_argument(
        "--theta1",
        type=float,
        default=15.0,
        help="Initial θ₁ perturbation in degrees (default: 15.0)",
    )
    parser.add_argument(
        "--theta2",
        type=float,
        default=-3.5,
        help="Initial θ₂ perturbation in degrees (default: -3.5)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Simulation duration in seconds (default: 10.0)",
    )

    args = parser.parse_args()

    run_ppo_viewer(
        model_path=args.model,
        initial_angles_deg=(args.theta1, args.theta2),
        duration=args.duration,
    )


if __name__ == "__main__":
    main()

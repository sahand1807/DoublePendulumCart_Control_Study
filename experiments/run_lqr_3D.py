"""
LQR Controller with MuJoCo 3D Viewer

Visualizes the LQR controller stabilizing the double pendulum in real-time.
"""
import mujoco
import mujoco_viewer
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from controllers.lqr_controller import create_lqr_controller

print("=" * 60)
print("LQR Controller with 3D MuJoCo Viewer")
print("=" * 60)
print("=" * 60)

# Load model
model = mujoco.MjModel.from_xml_path("env/double_pendulum_cart.xml")
data = mujoco.MjData(model)

# Create LQR controller
print("\nInitializing LQR controller...")
controller = create_lqr_controller()
print("  Controller ready!")

# Get max force from model (should be 20N based on ctrlrange)
actuator_id = 0
max_force = model.actuator_ctrlrange[actuator_id, 1]  # Should be 20
print(f"  Max force from model: ±{max_force}N")

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

# Reset data
mujoco.mj_resetData(model, data)

# Set initial state - small perturbation from upright
theta1_abs = np.pi + np.pi/8
theta2_abs = np.pi + np.pi/20

# Convert to MuJoCo convention
data.qpos[0] = 0.0  # cart at center
data.qpos[1] = theta1_abs
data.qpos[2] = theta2_abs - theta1_abs  # relative angle
data.qvel[:] = 0.0

# Forward kinematics
mujoco.mj_forward(model, data)

print(f"\nInitial state:")
print(f"  θ₁ = {np.degrees(theta1_abs):.2f}° (upright = 180°)")
print(f"  θ₂ = {np.degrees(theta2_abs):.2f}° (upright = 180°)")
print(f"\nStarting LQR control...\n")

# Create viewer
viewer = mujoco_viewer.MujocoViewer(model, data, hide_menus=True)

# Configure camera for better view
viewer.cam.azimuth = -90      # Side view
viewer.cam.elevation = -15   # Slightly from above
viewer.cam.distance = 8    # Distance from target (zoom out)
viewer.cam.lookat[0] = 0.0   # Look at x=0
viewer.cam.lookat[1] = 0.0   # Look at y=0
viewer.cam.lookat[2] = 0.3   # Look at z=0.3 (center of system)

# Simulation parameters
step = 0
print_interval = 50  # Print every 50 steps
dt = model.opt.timestep

# Statistics tracking
max_control_raw = 0.0        # Max LQR output (before clipping)
max_control_applied = 0.0    # Max actually applied (after clipping)
saturation_count = 0         # How many steps hit limits

# Settling detection parameters
settling_threshold = np.radians(3.0)  # 3 degrees = 0.0524 radians
settling_window = 100        # Must stay within threshold for 100 steps (1 second)
settled_counter = 0          # Counter for consecutive steps within threshold
settled_time = None          # Time when settled

print(f"Settling criteria:")
print(f"  Threshold: ±{np.degrees(settling_threshold):.1f}° for both angles")
print(f"  Must hold for: {settling_window} steps ({settling_window * dt:.1f}s)")
print()

while viewer.is_alive:
    # Get current state in absolute convention
    state = get_state_from_mujoco(data)
    
    # Compute LQR control (returns force in Newtons)
    u = controller.compute_control(state)
    
    # Track max control (raw output from LQR)
    max_control_raw = max(max_control_raw, abs(u))
    
    # Clip to actuator limits and apply to MuJoCo
    u_clipped = np.clip(u, -max_force, max_force)
    data.ctrl[0] = u_clipped
    
    # Track applied control and saturation
    max_control_applied = max(max_control_applied, abs(u_clipped))
    if abs(u) > max_force:
        saturation_count += 1
    
    # Step physics
    mujoco.mj_step(model, data)
    
    # Update viewer
    viewer.render()
    
    # Check settling condition
    # Both angles must be within threshold from upright (π radians)
    theta1_error = abs(state[1] - np.pi)
    theta2_error = abs(state[2] - np.pi)
    
    if theta1_error < settling_threshold and theta2_error < settling_threshold:
        # Within threshold - increment counter
        settled_counter += 1
        
        # If we've been within threshold for the full window, mark as settled
        if settled_counter >= settling_window and settled_time is None:
            settled_time = (step - settling_window + 1) * dt
            print(f"\n STABILIZED at t={settled_time:.2f}s!")
            print(f"   (Both angles stayed within ±{np.degrees(settling_threshold):.1f}° for {settling_window * dt:.1f}s)")
    else:
        # Outside threshold - reset counter
        settled_counter = 0
    
    # Print state occasionally
    if step % print_interval == 0:
        t = step * dt
        x, theta1, theta2, dx, dtheta1, dtheta2 = state
        
        theta1_error_deg = abs(np.degrees(theta1) - 180)
        theta2_error_deg = abs(np.degrees(theta2) - 180)
        
        status = "SETTLED" if settled_time is not None else f"settling... ({settled_counter}/{settling_window})"
        
        print(f"t={t:5.2f}s: "
              f"θ₁={np.degrees(theta1):6.2f}° (err={theta1_error_deg:5.2f}°), "
              f"θ₂={np.degrees(theta2):6.2f}° (err={theta2_error_deg:5.2f}°), "
              f"u={u:+7.2f}N → {u_clipped:+6.2f}N | {status}")
    
    step += 1
    
    # Optional: Stop after holding stable for 5 seconds
    # if settled_time is not None and (step * dt - settled_time) > 5.0:
    #     print("\nHeld stable for 5s, stopping...")
    #     break

# Close viewer
viewer.close()

print("\n" + "=" * 60)
print("Simulation Summary")
print("=" * 60)
print(f"Total simulation time: {step * dt:.2f}s")
print(f"Total steps: {step}")

print(f"\nControl Statistics:")
print(f"  Max LQR output (desired): {max_control_raw:.2f}N")
print(f"  Max applied (actual):     {max_control_applied:.2f}N (limit: ±{max_force}N)")
print(f"  Saturation events: {saturation_count}/{step} steps ({100*saturation_count/step:.1f}%)")

print(f"\nStabilization:")
if settled_time is not None:
    print(f"    Settling time: {settled_time:.2f}s")
    print(f"    (Time until both angles stayed within ±{np.degrees(settling_threshold):.1f}°")
    print(f"     for {settling_window} consecutive steps = {settling_window * dt:.1f}s)")
else:
    print(f"    Did not settle within threshold")
    print(f"    (Both angles must stay within ±{np.degrees(settling_threshold):.1f}° for {settling_window * dt:.1f}s)")

print("=" * 60)
print(" Viewer closed")
print("=" * 60)
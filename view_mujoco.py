"""
Simple 3D viewer - Start near upright, watch free fall

ANGLE CONVENTION (matching derivation):
- MuJoCo: qpos[1]=θ₁ (absolute), qpos[2]=θ₂_rel (relative to pole1)
- Our derivation: θ₁ (absolute), θ₂ (absolute from vertical)
- Upright: θ₁=π, θ₂=π in our convention
- In MuJoCo: qpos[1]=π, qpos[2]=0 (since θ₂_rel = θ₂ - θ₁ = π - π = 0)
"""
import mujoco
import mujoco_viewer
import numpy as np

print("=" * 60)
print("3D MuJoCo Viewer - Free Fall from Upright")
print("=" * 60)
print("\n🖱️  Controls:")
print("  - Left drag: Rotate camera")
print("  - Right drag: Pan camera")
print("  - Scroll: Zoom")
print("  - ESC: Close viewer")
print("\nStarting near upright position...")
print("No control applied - pure free fall dynamics")
print("=" * 60)

# Load model
model = mujoco.MjModel.from_xml_path("env/double_pendulum_cart.xml")
data = mujoco.MjData(model)

# Reset data
mujoco.mj_resetData(model, data)

# Set initial state - UPRIGHT with small perturbation
# Our derivation: θ₁ = π, θ₂ = π is upright
# MuJoCo needs: qpos[1] = θ₁, qpos[2] = θ₂ - θ₁

# Small perturbation from upright in absolute angles
theta1_abs = np.pi + 0.1  # Slightly away from upright
theta2_abs = np.pi + 0.05  # Slightly away from upright

# Convert to MuJoCo convention
data.qpos[0] = 0.0  # cart at center
data.qpos[1] = theta1_abs  # θ₁ (absolute)
data.qpos[2] = theta2_abs - theta1_abs  # θ₂_rel = θ₂ - θ₁
data.qvel[:] = 0.0  # zero velocities

# Forward kinematics
mujoco.mj_forward(model, data)

print(f"\nInitial conditions (absolute convention):")
print(f"  Cart position: {data.qpos[0]:.3f} m")
print(f"  θ₁: {theta1_abs:.3f} rad ({np.degrees(theta1_abs):.1f}°)")
print(f"  θ₂: {theta2_abs:.3f} rad ({np.degrees(theta2_abs):.1f}°)")
print(f"  Upright equilibrium: θ₁ = θ₂ = π = {np.pi:.3f} rad")
print(f"\nMuJoCo internal (for reference):")
print(f"  qpos[1] = {data.qpos[1]:.3f} (θ₁ absolute)")
print(f"  qpos[2] = {data.qpos[2]:.3f} (θ₂ relative)")
print(f"\nSystem parameters:")
print(f"  Cart mass: 1.0 kg")
print(f"  Link 1: mass=0.3kg, length=0.5m, COM at 0.25m")
print(f"  Link 2: mass=0.2kg, length=0.4m, COM at 0.2m")
print("\nLaunching viewer...\n")

# Create viewer
viewer = mujoco_viewer.MujocoViewer(model, data)

# Simulation loop
step = 0
print_interval = 100

while viewer.is_alive:
    # NO CONTROL - pure free fall
    data.ctrl[0] = 0.0
    
    # Step physics
    mujoco.mj_step(model, data)
    
    # Update viewer
    viewer.render()
    
    # Print state occasionally
    if step % print_interval == 0 and step > 0:
        t = step * model.opt.timestep
        x = data.qpos[0]
        
        # Convert to absolute angles for display
        theta1_abs = data.qpos[1]
        theta2_abs = data.qpos[1] + data.qpos[2]  # θ₂ = θ₁ + θ₂_rel
        
        print(f"t={t:5.2f}s: x={x:6.3f}m, "
              f"θ₁={np.degrees(theta1_abs):7.2f}°, "
              f"θ₂={np.degrees(theta2_abs):7.2f}°")
    
    step += 1

# Close viewer
viewer.close()

print("\n" + "=" * 60)
print("✓ Viewer closed")
print("=" * 60)
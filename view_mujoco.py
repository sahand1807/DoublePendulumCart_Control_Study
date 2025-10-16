"""
Simple 3D viewer - Start near upright, watch free fall
"""
import mujoco
import mujoco_viewer
import numpy as np

print("=" * 60)
print("3D MuJoCo Viewer - Free Fall from Upright")
print("=" * 60)
print("\n  Controls:")
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

# Reset data first
mujoco.mj_resetData(model, data)

# Now set initial state (close to upright)
# In your convention: theta = pi means upright (vertical up)
# Small perturbation from upright equilibrium
data.qpos[0] = 0.0           # cart at center
data.qpos[1] = np.pi + 0.5   # theta1 - slightly away from upright (pi)
data.qpos[2] = 0.5  # theta2
data.qvel[:] = 0.0           # zero velocities

# Forward kinematics to update visualization
mujoco.mj_forward(model, data)

print(f"\nInitial conditions:")
print(f"  Cart position: {data.qpos[0]:.3f} m")
print(f"  θ₁: {data.qpos[1]:.3f} rad ({np.degrees(data.qpos[1]):.1f}°)")
print(f"  θ₂: {data.qpos[2]:.3f} rad ({np.degrees(data.qpos[2]):.1f}°)")
print(f"  Upright equilibrium is at θ = π = {np.pi:.3f} rad")
print(f"\nSystem parameters:")
print(f"  Cart mass: 1.0 kg")
print(f"  Link 1: mass=0.3kg, length=0.5m, COM at 0.25m")
print(f"  Link 2: mass=0.2kg, length=0.4m, COM at 0.2m")
print("\nLaunching viewer...\n")

# Create viewer
viewer = mujoco_viewer.MujocoViewer(model, data)

# Simulation loop
step = 0
print_interval = 100  # Print every 100 steps

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
        th1 = data.qpos[1]
        th2 = data.qpos[2]
        
        # Calculate energy for monitoring
        # Potential energy (relative to cart level)
        PE = -0.3 * 9.81 * 0.25 * np.cos(th1) - 0.2 * 9.81 * (0.5 * np.cos(th1) + 0.2 * np.cos(th2))
        
        print(f"t={t:5.2f}s: x={x:6.3f}m, θ₁={np.degrees(th1):7.2f}°, "
              f"θ₂={np.degrees(th2):7.2f}°, PE={PE:7.3f}J")
    
    step += 1

# Close viewer
viewer.close()

print("\n" + "=" * 60)
print("✓ Viewer closed")
print("=" * 60)
"""
Single Pendulum Cart Test Script with Time Limit Wrapper

Demonstrates the TimeLimitWrapper which:
1. Disables angle-based termination (pendulum can fall without ending episode)
2. Removes angle limits (pendulum can rotate freely, multiple revolutions)
3. Only truncates after 10 seconds (configurable)
4. Accepts custom initial conditions

This is useful for:
- Learning swing-up behaviors
- Training robust recovery policies
- Exploring the full state space
- Allowing continuous rotation
"""
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from time_limit_wrapper import TimeLimitWrapper
import mujoco

# ==================== CONFIGURATION ====================

# Initial conditions: [cart_pos, cart_vel, pole_angle, pole_ang_vel]
# angle: 0 = upright, π = hanging down
INITIAL_CONDITIONS = {
    'cart_position': 0.0,      # meters (cart position)
    'cart_velocity': 0.0,      # m/s
    'pole_angle': np.pi/4,     # radians (0=upright, π=hanging down)
    'pole_angular_velocity': 0.0  # rad/s
}

# Episode duration
MAX_DURATION = 10.0  # seconds

# Control policy (simple example)
def control_policy(obs, step):
    """
    Define your control policy here.
    obs = [cart_pos, cart_vel, pole_angle, pole_ang_vel]

    Returns: action in [-1, 1]
    """
    # Example: No control (let it swing freely)
    return 0.0

    # Example: Alternating torque
    # return 0.5 * np.sin(step * 0.1)

    # Example: Proportional control
    # angle_error = obs[2]  # Distance from upright (0)
    # return -2.0 * angle_error  # P controller

# ==================== SETUP ====================

# Custom wrapper to zoom out the camera
class ZoomedOutWrapper(gym.Wrapper):
    def __init__(self, env, zoom_scale=2.0):
        super().__init__(env)
        self.zoom_scale = zoom_scale
        self._setup_camera()

    def _setup_camera(self):
        """Set up camera for better zoomed-out view"""
        if hasattr(self.env.unwrapped, 'model'):
            model = self.env.unwrapped.model
            # Increase the extent (field of view) to zoom out
            # This controls how much of the scene is visible
            if hasattr(model.vis, 'global_') and hasattr(model.vis.global_, 'extent'):
                # Store original extent
                self.original_extent = model.vis.global_.extent
                # Zoom out by increasing extent
                model.vis.global_.extent = self.original_extent * self.zoom_scale
                print(f"  Camera zoomed out: extent {self.original_extent:.1f} -> {model.vis.global_.extent:.1f}")

# Custom wrapper to set initial conditions
class InitialConditionWrapper(gym.Wrapper):
    def __init__(self, env, initial_state):
        super().__init__(env)
        self.initial_state = initial_state

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)

        # Set initial state in MuJoCo
        model = self.env.unwrapped.model
        data = self.env.unwrapped.data

        # InvertedPendulum-v5 has 2 joints: slider (cart) and hinge (pole)
        data.qpos[0] = self.initial_state['cart_position']
        data.qpos[1] = self.initial_state['pole_angle']
        data.qvel[0] = self.initial_state['cart_velocity']
        data.qvel[1] = self.initial_state['pole_angular_velocity']

        mujoco.mj_forward(model, data)

        # Get observation from the base environment
        obs = self.env.unwrapped._get_obs()
        return obs, info

# Create base environment with larger render size
base_env = gym.make("InvertedPendulum-v5", render_mode="rgb_array")

# Apply wrappers in order
env = ZoomedOutWrapper(base_env, zoom_scale=2.5)  # 2.5x zoom out
env = TimeLimitWrapper(env, max_duration=MAX_DURATION, unlimited_angle=True)
env = InitialConditionWrapper(env, INITIAL_CONDITIONS)

print("\n=== Single Pendulum Cart Test ===")
print(f"Initial conditions:")
print(f"  Cart position: {INITIAL_CONDITIONS['cart_position']:.3f} m")
print(f"  Cart velocity: {INITIAL_CONDITIONS['cart_velocity']:.3f} m/s")
print(f"  Pole angle: {INITIAL_CONDITIONS['pole_angle']:.3f} rad ({np.rad2deg(INITIAL_CONDITIONS['pole_angle']):.1f}°)")
print(f"  Pole ang. vel: {INITIAL_CONDITIONS['pole_angular_velocity']:.3f} rad/s")
print(f"Episode duration: {MAX_DURATION}s\n")

obs, info = env.reset()

# ==================== VISUALIZATION ====================

# Create figure with subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
ax1.set_title("Single Pendulum Cart (Zoomed Out View)", fontsize=12)
ax1.axis("off")
img = None

# Plot for tracking angle over time
ax2.set_title("State Tracking", fontsize=12)
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Angle (rad)", color='b')
ax2.tick_params(axis='y', labelcolor='b')
ax2.grid(True, alpha=0.3)

# Reference lines for angle
ax2.axhline(y=0, color='green', linestyle='-', linewidth=1.5, alpha=0.4, label='0° (Upright)')
ax2.axhline(y=np.pi/2, color='orange', linestyle='--', alpha=0.4, label='±90°')
ax2.axhline(y=-np.pi/2, color='orange', linestyle='--', alpha=0.4)
ax2.axhline(y=np.pi, color='red', linestyle=':', alpha=0.4, label='±180° (Hanging)')
ax2.axhline(y=-np.pi, color='red', linestyle=':', alpha=0.4)

# Twin axis for cart position
ax2_twin = ax2.twinx()
ax2_twin.set_ylabel('Cart Position (m)', color='purple')
ax2_twin.tick_params(axis='y', labelcolor='purple')

times = []
angles = []
cart_positions = []

step_count = 0

print("=== Starting Simulation ===\n")

# ==================== MAIN LOOP ====================

try:
    while True:
        # Get action from control policy
        action = control_policy(obs, step_count)
        action = np.clip(action, -1.0, 1.0)  # Ensure action is in valid range

        # Step environment
        obs, reward, terminated, truncated, info = env.step([action])
        frame = env.render()

        step_count += 1
        current_time = info.get('time_elapsed', 0)
        current_angle = obs[2]  # theta
        cart_pos = obs[0]  # x

        # Record data
        times.append(current_time)
        angles.append(current_angle)
        cart_positions.append(cart_pos)

        # Update visualization
        if img is None:
            img = ax1.imshow(frame)
        else:
            img.set_data(frame)

        # Update plots every 5 steps for smooth animation
        if step_count % 5 == 0:
            ax2.clear()
            ax2_twin.clear()

            # Redraw angle plot
            ax2.set_title(f"State Tracking (t={current_time:.1f}s)", fontsize=12)
            ax2.set_xlabel("Time (s)")
            ax2.set_ylabel("Angle (rad)", color='b')
            ax2.tick_params(axis='y', labelcolor='b')
            ax2.grid(True, alpha=0.3)

            # Reference lines
            ax2.axhline(y=0, color='green', linestyle='-', linewidth=1.5, alpha=0.4)
            ax2.axhline(y=np.pi/2, color='orange', linestyle='--', alpha=0.4)
            ax2.axhline(y=-np.pi/2, color='orange', linestyle='--', alpha=0.4)
            ax2.axhline(y=np.pi, color='red', linestyle=':', alpha=0.4)
            ax2.axhline(y=-np.pi, color='red', linestyle=':', alpha=0.4)

            # Plot angle
            ax2.plot(times, angles, 'b-', linewidth=2.5, label='Pole angle')
            ax2.legend(loc='upper left')
            ax2.set_xlim(0, MAX_DURATION)

            # Plot cart position on twin axis
            ax2_twin.set_ylabel('Cart Position (m)', color='purple')
            ax2_twin.tick_params(axis='y', labelcolor='purple')
            ax2_twin.plot(times, cart_positions, 'purple', linewidth=2, alpha=0.7, linestyle='--', label='Cart pos')
            ax2_twin.legend(loc='upper right')

        plt.pause(0.02)

        # Print status every 50 steps
        if step_count % 50 == 0:
            rotations = current_angle / (2 * np.pi)
            print(f"t={current_time:.2f}s | angle={current_angle:+.3f} rad ({rotations:+.2f} rot) | "
                  f"cart={cart_pos:+.3f}m | action={action:+.3f}")

        # Check if episode ended
        if terminated or truncated:
            duration = info.get('time_elapsed', 0)
            final_angle = obs[2]

            print(f"\n=== Episode Complete ===")
            print(f"Duration: {duration:.2f}s")
            print(f"Final state:")
            print(f"  Cart position: {obs[0]:+.3f} m")
            print(f"  Cart velocity: {obs[1]:+.3f} m/s")
            print(f"  Pole angle: {obs[2]:+.3f} rad ({np.rad2deg(obs[2]):+.1f}°)")
            print(f"  Pole ang. vel: {obs[3]:+.3f} rad/s")
            print(f"\nTerminated: {terminated} | Truncated: {truncated}")
            break

except KeyboardInterrupt:
    print("\n\nSimulation interrupted by user.")

env.close()

print("\n=== Simulation Complete ===\n")

# Keep window open
plt.show()
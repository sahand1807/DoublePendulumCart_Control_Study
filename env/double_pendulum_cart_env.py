"""
Double Pendulum Cart Environment using MuJoCo

IMPORTANT ANGLE CONVENTION:
- MuJoCo uses relative angles (hinge2 is relative to pole1)
- Our derivation uses absolute angles (both measured from vertical)
- This environment converts MuJoCo angles to absolute convention

State vector: [x, θ₁, θ₂, ẋ, θ̇₁, θ̇₂]
where θ₁ and θ₂ are ABSOLUTE angles from vertical (downward)
- θ = 0: hanging down
- θ = π: upright (inverted)
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import os


class DoublePendulumCartEnv(gym.Env):
    """
    Double pendulum on cart environment.
    
    State convention matches analytical derivation:
    - θ₁: absolute angle of first pendulum from vertical
    - θ₂: absolute angle of second pendulum from vertical
    - Upright equilibrium: [0, π, π, 0, 0, 0]
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}
    
    def __init__(self, render_mode=None, xml_path=None, terminate_on_fall=True, max_force=20.0):
        super().__init__()

        # Termination behavior
        self.terminate_on_fall = terminate_on_fall  # Set False for visualization
        
        # Load MuJoCo model
        if xml_path is None:
            xml_path = os.path.join(os.path.dirname(__file__), 
                                   "double_pendulum_cart.xml")
        
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # Rendering setup
        self.render_mode = render_mode
        self.renderer = None
        if render_mode == "human":
            self.renderer = mujoco.Renderer(self.model, height=480, width=640)
        
        # Action space: force on cart
        # Default: 20.0 for stabilization
        # Swing-up tasks should use higher values (e.g., 100.0)
        self.max_force = max_force
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        
        # Observation space: [x, θ₁, θ₂, ẋ, θ̇₁, θ̇₂]
        # Angles are absolute, so range is [0, 2π] but we allow wrapping
        # Extract cart limit from MuJoCo model (slider joint range)
        slider_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'slider')
        slider_range = self.model.jnt_range[slider_id]
        self.x_limit = float(abs(slider_range[1]))  # Assumes symmetric range [-x, +x]

        self.x_vel_limit = 5.0
        self.theta_vel_limit = 4 * np.pi
        
        obs_high = np.array([
            self.x_limit,
            2 * np.pi,
            2 * np.pi,
            self.x_vel_limit,
            self.theta_vel_limit,
            self.theta_vel_limit
        ], dtype=np.float32)
        
        self.observation_space = spaces.Box(
            low=-obs_high, high=obs_high, dtype=np.float32
        )
        
        # Simulation parameters
        self.dt = self.model.opt.timestep
        self.frame_skip = 1  # CRITICAL: Reduce frame skip for better control
        
        # Episode parameters
        self.max_episode_steps = 2000  # Longer episodes to see full dynamics
        self.current_step = 0
        
    def _mujoco_to_absolute_angles(self, qpos_mujoco):
        """
        Convert MuJoCo angles (relative) to absolute angles from vertical.

        MuJoCo convention:
        - qpos[1] = θ₁ (absolute angle of pole1 from vertical)
        - qpos[2] = θ₂_rel (angle of pole2 relative to pole1)

        Our convention (from derivation):
        - θ₁ (absolute angle of pole1 from vertical)
        - θ₂ (absolute angle of pole2 from vertical)

        Conversion: θ₂_abs = θ₁ + θ₂_rel

        IMPORTANT: Angles are wrapped to [-π, π] to prevent accumulation.
        """
        x = qpos_mujoco[0]
        theta1_abs = qpos_mujoco[1]
        theta2_rel = qpos_mujoco[2]

        # Convert to absolute angle
        theta2_abs = theta1_abs + theta2_rel

        # Wrap angles to [-π, π] to prevent unbounded accumulation
        # This is critical for energy calculations and observations
        theta1_abs = np.arctan2(np.sin(theta1_abs), np.cos(theta1_abs))
        theta2_abs = np.arctan2(np.sin(theta2_abs), np.cos(theta2_abs))

        return x, theta1_abs, theta2_abs
    
    def _absolute_to_mujoco_angles(self, x, theta1_abs, theta2_abs):
        """
        Convert absolute angles to MuJoCo convention.
        
        θ₂_rel = θ₂_abs - θ₁
        """
        theta2_rel = theta2_abs - theta1_abs
        return np.array([x, theta1_abs, theta2_rel])
    
    def _get_obs(self):
        """Extract observation with absolute angle convention."""
        # Get MuJoCo state
        x, theta1_abs, theta2_abs = self._mujoco_to_absolute_angles(self.data.qpos)
        
        # Velocities (need conversion too!)
        dx = self.data.qvel[0]
        dtheta1 = self.data.qvel[1]
        dtheta2_rel = self.data.qvel[2]
        
        # Convert angular velocity: ω₂_abs = ω₁ + ω₂_rel
        dtheta2_abs = dtheta1 + dtheta2_rel
        
        # Return state in absolute convention
        obs = np.array([x, theta1_abs, theta2_abs, dx, dtheta1, dtheta2_abs], 
                      dtype=np.float32)
        return obs
    
    def _get_info(self):
        """
        Additional information for logging.

        IMPORTANT: MuJoCo geometry convention:
        - θ=0: UPRIGHT (rods point +z, gravity is -z)
        - θ=π: HANGING DOWN

        For swing-up task: start at θ=π (hanging), goal is θ=0 (upright)
        """
        obs = self._get_obs()
        return {
            "cart_position": obs[0],
            "pole1_angle": obs[1],
            "pole2_angle": obs[2],
            "upright_error": np.abs(obs[1]) + np.abs(obs[2])  # Distance from θ=0 (upright)
        }
    
    def reset(self, seed=None, options=None):
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        # Reset MuJoCo simulation
        mujoco.mj_resetData(self.model, self.data)
        
        # Set initial state
        if options is not None and "initial_state" in options:
            # User provides state in absolute convention [x, θ₁, θ₂, ẋ, θ̇₁, θ̇₂]
            initial_state = options["initial_state"]
            x = initial_state[0]
            theta1_abs = initial_state[1]
            theta2_abs = initial_state[2]
            dx = initial_state[3]
            dtheta1 = initial_state[4]
            dtheta2_abs = initial_state[5]
            
            # Convert to MuJoCo convention
            self.data.qpos[:] = self._absolute_to_mujoco_angles(x, theta1_abs, theta2_abs)
            
            # Convert velocities
            dtheta2_rel = dtheta2_abs - dtheta1
            self.data.qvel[:] = [dx, dtheta1, dtheta2_rel]
        else:
            # Random initialization near upright (θ=0, 0 in MuJoCo geometry)
            # IMPORTANT: MuJoCo has θ=0 as upright (rods point +z, gravity -z)
            self.data.qpos[0] = self.np_random.uniform(-0.1, 0.1)  # x
            theta1 = self.np_random.uniform(-0.1, 0.1)  # Near θ=0 (upright)
            theta2 = self.np_random.uniform(-0.1, 0.1)  # Near θ=0 (upright)

            # Convert to MuJoCo
            self.data.qpos[:] = self._absolute_to_mujoco_angles(
                self.data.qpos[0], theta1, theta2
            )

            self.data.qvel[:] = self.np_random.uniform(-0.05, 0.05, size=3)
        
        mujoco.mj_forward(self.model, self.data)
        
        self.current_step = 0
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action):
        """Execute one time step."""
        # Handle both scalar and array actions (for compatibility with VecEnv)
        if np.isscalar(action):
            force = np.clip(action, -1.0, 1.0) * self.max_force
        else:
            force = np.clip(action[0], -1.0, 1.0) * self.max_force
        self.data.ctrl[0] = force
        
        # Step physics simulation
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        
        # Get observation (with absolute angles)
        observation = self._get_obs()
        info = self._get_info()
        
        # Compute reward
        reward = self._compute_reward(observation, action)
        
        # Check termination
        self.current_step += 1
        terminated = self._is_terminated(observation)
        truncated = self.current_step >= self.max_episode_steps
        
        return observation, reward, terminated, truncated, info
    
    def _compute_reward(self, obs, action):
        """
        LQR-inspired reward function for stabilization with minimal cart oscillation.

        IMPORTANT: MuJoCo geometry convention:
        - θ=0: UPRIGHT equilibrium (rods point +z, gravity is -z)
        - θ=π: HANGING DOWN

        Upright equilibrium: θ₁ = 0, θ₂ = 0

        Priority structure matching LQR Q matrix:
        - Angles (highest priority): ~100 weight
        - Cart position: ~1.0 weight
        - Angular velocities: ~10 weight
        - Cart velocity: ~0.1 weight

        This ensures minimal cart drift while maintaining robust stabilization.
        """
        x, theta1, theta2, dx, dtheta1, dtheta2 = obs
        # Handle both scalar and array actions (for compatibility with VecEnv)
        if np.isscalar(action):
            force = action * self.max_force
        else:
            force = action[0] * self.max_force

        # Angle errors from upright (θ=0 in MuJoCo geometry)
        theta1_error = theta1  # Distance from 0
        theta2_error = theta2

        # Wrap to [-π, π]
        theta1_error = np.arctan2(np.sin(theta1_error), np.cos(theta1_error))
        theta2_error = np.arctan2(np.sin(theta2_error), np.cos(theta2_error))

        # Alive bonus: encourages staying upright
        alive_bonus = 10.0

        # LQR-inspired cost structure
        # Primary: Angle errors (highest priority, effective weight ~1.0 relative to alive bonus)
        angle_cost = theta1_error**2 + theta2_error**2

        # Secondary: Cart position (increased from 0.1 to 0.5 to reduce oscillation)
        # Matches LQR ratio of ~100:1 (angles:position)
        position_cost = 0.5 * x**2

        # Tertiary: Velocities (separated for better control)
        # Cart velocity: low penalty (0.01, matching LQR's conservative approach)
        # Angular velocities: moderate penalty (0.1, matching LQR's 10:100 ratio)
        cart_velocity_cost = 0.01 * dx**2
        angular_velocity_cost = 0.1 * (dtheta1**2 + dtheta2**2)

        # Control effort: small penalty for smooth control
        control_cost = 0.001 * force**2

        # Total reward with LQR-inspired priority structure
        # Perfect upright at center: angle_cost ≈ 0, position_cost ≈ 0, total ≈ +10
        # At x=0.5m: position_cost = 0.125 (moderate penalty)
        # At x=1.0m: position_cost = 0.5 (significant penalty)
        reward = alive_bonus - (angle_cost + position_cost + cart_velocity_cost
                                + angular_velocity_cost + control_cost)

        return reward
    
    def _is_terminated(self, obs):
        """
        Check if episode should terminate.

        IMPORTANT: MuJoCo geometry convention:
        - θ=0: UPRIGHT (rods point +z)
        - θ=π: HANGING DOWN

        For stabilization tasks, we want to stay near θ=0 (upright).
        For swing-up tasks, terminate_on_fall should be False.
        """
        x, theta1, theta2 = obs[0], obs[1], obs[2]

        # Always terminate if cart goes out of bounds
        if abs(x) > self.x_limit:
            return True

        # Optionally terminate on large angle deviations (for stabilization tasks)
        if self.terminate_on_fall:
            # Distance from upright (θ=0)
            theta1_error = abs(theta1)
            theta2_error = abs(theta2)

            # Wrap errors to [0, π]
            theta1_error = min(theta1_error, 2*np.pi - theta1_error)
            theta2_error = min(theta2_error, 2*np.pi - theta2_error)

            # Terminate if more than 90° from upright (fallen)
            if theta1_error > np.pi/2 or theta2_error > np.pi/2:
                return True

        return False
    
    def render(self):
        """Render the environment."""
        if self.render_mode == "human":
            if self.renderer is None:
                self.renderer = mujoco.Renderer(self.model, 
                                               height=480, width=640)
            self.renderer.update_scene(self.data)
            return self.renderer.render()
        return None
    
    def close(self):
        """Clean up resources."""
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None


# Register environment
gym.register(
    id='DoublePendulumCart-v0',
    entry_point='env.double_pendulum_cart_env:DoublePendulumCartEnv',
    max_episode_steps=2000,  # Longer episodes
)
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
    
    def __init__(self, render_mode=None, xml_path=None, terminate_on_fall=True):
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
        self.max_force = 20.0
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        
        # Observation space: [x, θ₁, θ₂, ẋ, θ̇₁, θ̇₂]
        # Angles are absolute, so range is [0, 2π] but we allow wrapping
        self.x_limit = 2.0
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
        """
        x = qpos_mujoco[0]
        theta1_abs = qpos_mujoco[1]
        theta2_rel = qpos_mujoco[2]
        
        # Convert to absolute angle
        theta2_abs = theta1_abs + theta2_rel
        
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
        """Additional information for logging."""
        obs = self._get_obs()
        return {
            "cart_position": obs[0],
            "pole1_angle": obs[1],
            "pole2_angle": obs[2],
            "upright_error": np.abs(obs[1] - np.pi) + np.abs(obs[2] - np.pi)
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
            # Random initialization near upright (π, π)
            self.data.qpos[0] = self.np_random.uniform(-0.1, 0.1)  # x
            theta1 = np.pi + self.np_random.uniform(-0.1, 0.1)
            theta2 = np.pi + self.np_random.uniform(-0.1, 0.1)
            
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
        Reward function for stabilization at upright position.
        Upright equilibrium: θ₁ = π, θ₂ = π

        Based on literature (PLOS ONE 2023):
        - Primary: cos(θ) reward for angle stabilization
        - Secondary: Cart position penalty to prevent drift
        - Minimal angular velocity penalty for smoothness
        - NO cart velocity penalty (needed for stabilization!)

        Reference: r = cos(θ) - (x/x₀)²
        """
        x, theta1, theta2, dx, dtheta1, dtheta2 = obs
        # Handle both scalar and array actions (for compatibility with VecEnv)
        if np.isscalar(action):
            force = action * self.max_force
        else:
            force = action[0] * self.max_force

        # Angle errors from upright (π)
        theta1_error = theta1 - np.pi
        theta2_error = theta2 - np.pi

        # Wrap to [-π, π]
        theta1_error = np.arctan2(np.sin(theta1_error), np.cos(theta1_error))
        theta2_error = np.arctan2(np.sin(theta2_error), np.cos(theta2_error))

        # Angle reward using cosine (literature-based approach)
        # cos(0) = 1 when perfectly upright, cos(π) = -1 when inverted
        angle_reward = np.cos(theta1_error) + np.cos(theta2_error)
        # Scale to [0, 10] range: (2 to -2) -> (10 to 0)
        angle_reward = 5.0 * (angle_reward + 2.0) / 2.0

        # Cart position penalty: quadratic, normalized by x_limit
        # Scaled to be significant but not dominating angle reward
        # At x=0: penalty=0, At x=x_limit=2.0m: penalty=2.0
        position_penalty = 2.0 * (x / self.x_limit)**2

        # Angular velocity penalty: very small, only for smoothness
        # Literature uses ~0.001 for angular velocities
        angular_velocity_penalty = 0.001 * (dtheta1**2 + dtheta2**2)

        # Control cost: small penalty for large forces
        control_penalty = 0.001 * force**2

        # NO cart velocity penalty - cart must be free to move for stabilization!
        # The position penalty prevents unbounded drift without constraining dynamics

        # Total reward
        # Perfect upright at center: angle_reward ≈ 10, penalties ≈ 0, total ≈ +10
        # Cart at x=1.0m: position_penalty = 0.5 (moderate)
        # Cart at x=2.0m: position_penalty = 2.0 (significant)
        reward = angle_reward - (position_penalty + angular_velocity_penalty + control_penalty)

        return reward
    
    def _is_terminated(self, obs):
        """Check if episode should terminate."""
        x, theta1, theta2 = obs[0], obs[1], obs[2]
        
        # Always terminate if cart goes out of bounds
        if abs(x) > self.x_limit:
            return True
        
        # Optionally terminate on large angle deviations (for control tasks)
        if self.terminate_on_fall:
            theta1_error = abs(theta1 - np.pi)
            theta2_error = abs(theta2 - np.pi)
            
            # Wrap errors to [0, π]
            theta1_error = min(theta1_error, 2*np.pi - theta1_error)
            theta2_error = min(theta2_error, 2*np.pi - theta2_error)
            
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
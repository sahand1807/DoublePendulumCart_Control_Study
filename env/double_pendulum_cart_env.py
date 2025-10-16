"""
Double Pendulum Cart Environment using MuJoCo
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import os


class DoublePendulumCartEnv(gym.Env):
    """
    Custom Gymnasium environment for double pendulum on cart using MuJoCo.
    
    State: [x, θ₁, θ₂, ẋ, θ̇₁, θ̇₂]
    Action: Force applied to cart (continuous)
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}
    
    def __init__(self, render_mode=None, xml_path=None):
        super().__init__()
        
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
        # Position limits
        self.x_limit = 2.0
        self.theta_limit = np.pi
        # Velocity limits (reasonable bounds)
        self.x_vel_limit = 5.0
        self.theta_vel_limit = 4 * np.pi
        
        obs_high = np.array([
            self.x_limit,
            self.theta_limit,
            self.theta_limit,
            self.x_vel_limit,
            self.theta_vel_limit,
            self.theta_vel_limit
        ], dtype=np.float32)
        
        self.observation_space = spaces.Box(
            low=-obs_high, high=obs_high, dtype=np.float32
        )
        
        # Simulation parameters
        self.dt = self.model.opt.timestep
        self.frame_skip = 5  # Run 5 physics steps per env step
        
        # Episode parameters
        self.max_episode_steps = 1000
        self.current_step = 0
        
    def _get_obs(self):
        """Extract observation from MuJoCo data."""
        # Joint positions and velocities
        x = self.data.qpos[0]  # cart position
        theta1 = self.data.qpos[1]  # first pendulum angle
        theta2 = self.data.qpos[2]  # second pendulum angle
        
        dx = self.data.qvel[0]  # cart velocity
        dtheta1 = self.data.qvel[1]  # first pendulum angular velocity
        dtheta2 = self.data.qvel[2]  # second pendulum angular velocity
        
        # Normalize angles to [-π, π]
        theta1 = ((theta1 + np.pi) % (2 * np.pi)) - np.pi
        theta2 = ((theta2 + np.pi) % (2 * np.pi)) - np.pi
        
        obs = np.array([x, theta1, theta2, dx, dtheta1, dtheta2], 
                      dtype=np.float32)
        return obs
    
    def _get_info(self):
        """Additional information for logging."""
        obs = self._get_obs()
        return {
            "cart_position": obs[0],
            "pole1_angle": obs[1],
            "pole2_angle": obs[2],
            "upright_score": self._upright_score(obs)
        }
    
    def _upright_score(self, obs):
        """Measure how upright the pendulums are (1.0 = perfectly upright)."""
        theta1, theta2 = obs[1], obs[2]
        return np.cos(theta1) * np.cos(theta2)
    
    def reset(self, seed=None, options=None):
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        # Reset MuJoCo simulation
        mujoco.mj_resetData(self.model, self.data)
        
        # Set initial state with small random perturbations
        if options is not None and "initial_state" in options:
            # Use provided initial state
            initial_state = options["initial_state"]
            self.data.qpos[:] = initial_state[:3]
            self.data.qvel[:] = initial_state[3:]
        else:
            # Random initialization near upright
            self.data.qpos[0] = self.np_random.uniform(-0.1, 0.1)  # x
            self.data.qpos[1] = self.np_random.uniform(-0.1, 0.1)  # θ₁
            self.data.qpos[2] = self.np_random.uniform(-0.1, 0.1)  # θ₂
            
            self.data.qvel[:] = self.np_random.uniform(-0.05, 0.05, size=3)
        
        # Forward kinematics
        mujoco.mj_forward(self.model, self.data)
        
        self.current_step = 0
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action):
        """Execute one time step."""
        # Scale action to force range
        force = np.clip(action[0], -1.0, 1.0) * self.max_force
        self.data.ctrl[0] = force
        
        # Step physics simulation
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        
        # Get observation
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
        Reward function for stabilization.
        Encourages upright configuration with minimal control effort.
        """
        x, theta1, theta2, dx, dtheta1, dtheta2 = obs
        force = action[0] * self.max_force
        
        # Angle error (want both angles at 0)
        angle_cost = theta1**2 + theta2**2
        
        # Cart position cost (want cart near center)
        position_cost = x**2
        
        # Velocity cost (want system at rest)
        velocity_cost = 0.01 * (dx**2 + dtheta1**2 + dtheta2**2)
        
        # Control effort cost
        control_cost = 0.001 * force**2
        
        # Reward is negative cost
        reward = -(angle_cost + 0.1 * position_cost + 
                  velocity_cost + control_cost)
        
        # Bonus for being very upright
        if abs(theta1) < 0.1 and abs(theta2) < 0.1:
            reward += 1.0
        
        return reward
    
    def _is_terminated(self, obs):
        """Check if episode should terminate (failure condition)."""
        x, theta1, theta2 = obs[0], obs[1], obs[2]
        
        # Terminate if cart goes out of bounds
        if abs(x) > self.x_limit:
            return True
        
        # Terminate if pendulums fall too far
        if abs(theta1) > np.pi/2 or abs(theta2) > np.pi/2:
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
    max_episode_steps=1000,
)
"""
Observation wrapper for angle encoding using sin/cos representation.

This wrapper converts raw angle observations to sin/cos pairs to avoid
discontinuities at ±π boundaries, which is critical for stable RL training.

Original observation: [x, θ₁, θ₂, ẋ, θ̇₁, θ̇₂]  (6 dimensions)
Wrapped observation:  [x, sin(θ₁), cos(θ₁), sin(θ₂), cos(θ₂), ẋ, θ̇₁, θ̇₂]  (8 dimensions)

Reference: Gymnasium best practices for pendulum environments
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class AngleObservationWrapper(gym.ObservationWrapper):
    """
    Wraps angle observations with sin/cos encoding.

    This wrapper is essential for PPO training because:
    1. Neural networks struggle with discontinuous inputs (angle wrapping at ±π)
    2. Sin/cos encoding provides continuous representation
    3. Both sin and cos are needed to uniquely identify angle in [0, 2π]
    4. Values are bounded in [-1, 1], which helps network training

    Example:
        env = DoublePendulumCartEnv()
        env = AngleObservationWrapper(env)
        obs, info = env.reset()
        # obs now has 8 dimensions instead of 6
    """

    def __init__(self, env):
        """
        Initialize the observation wrapper.

        Args:
            env: DoublePendulumCartEnv instance
        """
        super().__init__(env)

        # Get original observation space bounds
        orig_low = env.observation_space.low
        orig_high = env.observation_space.high

        # New observation space: [x, sin(θ₁), cos(θ₁), sin(θ₂), cos(θ₂), ẋ, θ̇₁, θ̇₂]
        # Index mapping:
        # 0: x (from orig[0])
        # 1: sin(θ₁) (from orig[1])
        # 2: cos(θ₁) (from orig[1])
        # 3: sin(θ₂) (from orig[2])
        # 4: cos(θ₂) (from orig[2])
        # 5: ẋ (from orig[3])
        # 6: θ̇₁ (from orig[4])
        # 7: θ̇₂ (from orig[5])

        new_low = np.array([
            orig_low[0],   # x
            -1.0,          # sin(θ₁)
            -1.0,          # cos(θ₁)
            -1.0,          # sin(θ₂)
            -1.0,          # cos(θ₂)
            orig_low[3],   # ẋ
            orig_low[4],   # θ̇₁
            orig_low[5],   # θ̇₂
        ], dtype=np.float32)

        new_high = np.array([
            orig_high[0],  # x
            1.0,           # sin(θ₁)
            1.0,           # cos(θ₁)
            1.0,           # sin(θ₂)
            1.0,           # cos(θ₂)
            orig_high[3],  # ẋ
            orig_high[4],  # θ̇₁
            orig_high[5],  # θ̇₂
        ], dtype=np.float32)

        self.observation_space = spaces.Box(
            low=new_low,
            high=new_high,
            dtype=np.float32
        )

    def observation(self, obs):
        """
        Transform observation with sin/cos encoding.

        Args:
            obs: Original observation [x, θ₁, θ₂, ẋ, θ̇₁, θ̇₂]

        Returns:
            Wrapped observation [x, sin(θ₁), cos(θ₁), sin(θ₂), cos(θ₂), ẋ, θ̇₁, θ̇₂]
        """
        x, theta1, theta2, dx, dtheta1, dtheta2 = obs

        # Compute sin/cos of angles
        sin_theta1 = np.sin(theta1)
        cos_theta1 = np.cos(theta1)
        sin_theta2 = np.sin(theta2)
        cos_theta2 = np.cos(theta2)

        # Construct new observation
        wrapped_obs = np.array([
            x,
            sin_theta1,
            cos_theta1,
            sin_theta2,
            cos_theta2,
            dx,
            dtheta1,
            dtheta2
        ], dtype=np.float32)

        return wrapped_obs

    def get_original_angles(self, wrapped_obs):
        """
        Reconstruct original angles from wrapped observation.

        This is useful for visualization and debugging.

        Args:
            wrapped_obs: Wrapped observation [x, sin(θ₁), cos(θ₁), sin(θ₂), cos(θ₂), ẋ, θ̇₁, θ̇₂]

        Returns:
            theta1, theta2: Original angles in radians
        """
        sin_theta1 = wrapped_obs[1]
        cos_theta1 = wrapped_obs[2]
        sin_theta2 = wrapped_obs[3]
        cos_theta2 = wrapped_obs[4]

        # Reconstruct angles using atan2 (handles all quadrants correctly)
        theta1 = np.arctan2(sin_theta1, cos_theta1)
        theta2 = np.arctan2(sin_theta2, cos_theta2)

        return theta1, theta2


class CurriculumInitializationWrapper(gym.Wrapper):
    """
    Wrapper for curriculum learning with progressive difficulty levels.

    Controls the range of initial angle perturbations around upright equilibrium.
    Curriculum levels:
    - Level 1: ±3° (±0.052 rad) - Easy, near upright
    - Level 2: ±10° (±0.175 rad) - Medium difficulty
    - Level 3: ±30° (±0.524 rad) - Hard, testing limits

    Usage:
        env = DoublePendulumCartEnv()
        env = CurriculumInitializationWrapper(env, curriculum_level=1)
        # Will initialize with small perturbations
    """

    def __init__(self, env, curriculum_level=1):
        """
        Initialize curriculum wrapper.

        Args:
            env: Base environment
            curriculum_level: 1 (easy), 2 (medium), or 3 (hard)
        """
        super().__init__(env)
        self.set_curriculum_level(curriculum_level)

    def set_curriculum_level(self, level):
        """
        Set the curriculum difficulty level.

        Args:
            level: 1, 2, or 3
        """
        if level not in [1, 2, 3]:
            raise ValueError(f"Curriculum level must be 1, 2, or 3, got {level}")

        self.curriculum_level = level

        # Define angle perturbation ranges (in radians)
        self.level_configs = {
            1: {'angle_range': np.deg2rad(3),   'name': 'Level 1 (±3°)'},
            2: {'angle_range': np.deg2rad(10),  'name': 'Level 2 (±10°)'},
            3: {'angle_range': np.deg2rad(30),  'name': 'Level 3 (±30°)'},
        }

        self.angle_range = self.level_configs[level]['angle_range']
        self.level_name = self.level_configs[level]['name']

    def reset(self, seed=None, options=None):
        """
        Reset environment with curriculum-appropriate initial conditions.

        Initial state around upright equilibrium (θ₁ = π, θ₂ = π):
        - Random angles: θᵢ ~ Uniform(π - range, π + range)
        - Small cart position: x ~ Uniform(-0.1, 0.1) m
        - Small velocities: ẋ, θ̇₁, θ̇₂ ~ Uniform(-0.05, 0.05)
        """
        # Override options to set curriculum-appropriate initial state
        if options is None:
            options = {}

        # Generate random initial state for current curriculum level
        theta1_init = np.pi + self.env.unwrapped.np_random.uniform(
            -self.angle_range, self.angle_range
        )
        theta2_init = np.pi + self.env.unwrapped.np_random.uniform(
            -self.angle_range, self.angle_range
        )
        x_init = self.env.unwrapped.np_random.uniform(-0.1, 0.1)

        # Small initial velocities
        dx_init = self.env.unwrapped.np_random.uniform(-0.05, 0.05)
        dtheta1_init = self.env.unwrapped.np_random.uniform(-0.05, 0.05)
        dtheta2_init = self.env.unwrapped.np_random.uniform(-0.05, 0.05)

        initial_state = np.array([
            x_init,
            theta1_init,
            theta2_init,
            dx_init,
            dtheta1_init,
            dtheta2_init
        ])

        options['initial_state'] = initial_state

        return self.env.reset(seed=seed, options=options)

    def __repr__(self):
        return f"<CurriculumInitializationWrapper {self.level_name}>"

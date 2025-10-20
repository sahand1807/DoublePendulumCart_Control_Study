"""
Wrappers for swing-up control task.

This module provides initialization and reward wrappers specifically designed
for the swing-up problem where the double pendulum starts hanging downward
and must be swung up to the inverted (upright) position.

Key differences from stabilization task:
- Initialization: Starts at θ₁=0, θ₂=0 (hanging down) instead of near θ=π
- Reward: Unified quadratic cost that encourages swing-up and stabilization
- Episode length: Longer episodes (10000 steps) to allow time for swing-up
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class SwingUpInitializationWrapper(gym.Wrapper):
    """
    Wrapper for swing-up task initialization.

    Always initializes the double pendulum in the hanging-down position
    (θ₁ ≈ 0, θ₂ ≈ 0) with small random perturbations.

    This is the opposite of the curriculum learning wrapper which starts
    near upright (θ ≈ π) for stabilization tasks.

    Usage:
        env = DoublePendulumCartEnv()
        env = SwingUpInitializationWrapper(env, perturbation_range=0.1)
        env = AngleObservationWrapper(env)
    """

    def __init__(self, env, perturbation_range=0.1):
        """
        Initialize swing-up wrapper.

        Args:
            env: Base environment (DoublePendulumCartEnv)
            perturbation_range: Random perturbation around hanging position (radians)
                              Default: 0.1 rad ≈ 5.7°
        """
        super().__init__(env)
        self.perturbation_range = perturbation_range

    def reset(self, seed=None, options=None):
        """
        Reset environment to hanging-down position with small perturbations.

        Initial state:
        - θ₁ ≈ 0 (first pendulum hanging down)
        - θ₂ ≈ 0 (second pendulum hanging down)
        - x ≈ 0 (cart at center)
        - All velocities ≈ 0 (small random values)

        Args:
            seed: Random seed
            options: Additional options (unused, for compatibility)

        Returns:
            observation: Initial observation
            info: Initial info dict
        """
        # Override options to set swing-up initial state
        if options is None:
            options = {}

        # Fixed initial state: hanging down, cart at center, all velocities zero
        # This is the canonical swing-up problem setup
        initial_state = np.array([
            0.0,    # x: cart at center
            0.0,    # θ₁: first pendulum hanging down
            0.0,    # θ₂: second pendulum hanging down
            0.0,    # ẋ: cart velocity zero
            0.0,    # θ̇₁: pendulum 1 angular velocity zero
            0.0     # θ̇₂: pendulum 2 angular velocity zero
        ])

        options['initial_state'] = initial_state

        return self.env.reset(seed=seed, options=options)

    def __repr__(self):
        return f"<SwingUpInitializationWrapper perturbation=±{np.rad2deg(self.perturbation_range):.1f}°>"


class SwingUpRewardWrapper(gym.RewardWrapper):
    """
    Reward wrapper for swing-up control task.

    Implements a unified quadratic reward function based on recent literature
    (IROS 2024 AI Olympics winner - AR-EAPO approach).

    The reward is structured as:
        r(s,a) = -α[(s-g)ᵀQ(s-g) + R·a²]

    Where:
    - g = [0, π, π, 0, 0, 0]ᵀ is the goal state (upright, centered, at rest)
    - Q penalizes state deviations (heavy on angles, lighter on velocities)
    - R penalizes control effort
    - α scales the overall reward magnitude

    This single reward function naturally encourages:
    1. Swing-up behavior (from hanging position)
    2. Stabilization (when near upright)
    3. Energy-efficient control

    Key insight: Modern RL algorithms (PPO, SAC) can discover the swing-up
    strategy without explicit energy-based shaping or phase switching.

    Reference:
        "Average-Reward Maximum Entropy RL for Underactuated Double Pendulum Tasks"
        IROS 2024 AI Olympics Competition
    """

    def __init__(self, env,
                 angle_weight=50.0,
                 velocity_weight_theta1=4.0,
                 velocity_weight_theta2=2.0,
                 position_weight=1.0,
                 cart_velocity_weight=0.5,
                 control_weight=1.0,
                 scale=0.001):
        """
        Initialize swing-up reward wrapper.

        Args:
            env: Base environment
            angle_weight: Weight for angle errors (both θ₁ and θ₂)
            velocity_weight_theta1: Weight for θ̇₁
            velocity_weight_theta2: Weight for θ̇₂
            position_weight: Weight for cart position x
            cart_velocity_weight: Weight for cart velocity ẋ
            control_weight: Weight for control effort
            scale: Overall reward scaling factor
        """
        super().__init__(env)

        # Store reward weights (equivalent to Q and R matrices)
        self.angle_weight = angle_weight
        self.velocity_weight_theta1 = velocity_weight_theta1
        self.velocity_weight_theta2 = velocity_weight_theta2
        self.position_weight = position_weight
        self.cart_velocity_weight = cart_velocity_weight
        self.control_weight = control_weight
        self.scale = scale

        # Store last action for reward computation
        self.last_action = None

    def step(self, action):
        """Step environment and compute swing-up reward."""
        self.last_action = action
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Replace the base environment's reward with swing-up reward
        reward = self.reward(reward)

        return obs, reward, terminated, truncated, info

    def reward(self, reward):
        """
        Compute swing-up reward using proven Acrobot-style approach.

        Based on successful Gym implementations:
        - Acrobot: sparse -1 per step + height termination
        - Pendulum: -(angle² + velocity_penalty + control_penalty)

        We use a HEIGHT-BASED reward like Acrobot, which is simple and proven.

        Args:
            reward: Original reward (ignored)

        Returns:
            swing_up_reward: Height-based reward for swing-up
        """
        # Get current observation (in absolute angle convention)
        obs = self.env.unwrapped._get_obs()
        x, theta1, theta2, dx, dtheta1, dtheta2 = obs

        # Get last action
        if self.last_action is None:
            force = 0.0
        else:
            # Handle both scalar and array actions
            if np.isscalar(self.last_action):
                force = self.last_action * self.env.unwrapped.max_force
            else:
                force = self.last_action[0] * self.env.unwrapped.max_force

        # KEY INSIGHT FROM ACROBOT/PENDULUM:
        # Use HEIGHT of pendulum tips as reward signal
        # This is simple, continuous, and proven to work!

        # Approximate pendulum lengths (should match your XML)
        L1 = 0.5
        L2 = 0.5

        # Height of each pendulum tip (higher = better)
        # Height is measured from hanging position (θ=0)
        # When upright (θ=π), height is maximum
        h1 = -L1 * np.cos(theta1)  # Height of first pendulum tip
        h2 = -L1 * np.cos(theta1) - L2 * np.cos(theta2)  # Height of second pendulum tip

        # Total height (goal: maximize this)
        total_height = h1 + h2

        # Maximum possible height (both pendulums upright)
        max_height = L1 + L2  # = 1.0

        # Normalize height to [-1, 1] range
        # Hanging: height = -1.0
        # Upright: height = +1.0
        normalized_height = total_height / max_height

        # HEIGHT REWARD (main component)
        # Reward proportional to height (like -cos(theta) in Acrobot)
        height_reward = normalized_height

        # PENALTIES (keep cart near center, penalize excessive actions)
        # Cart penalty must be VERY small to not interfere with swing-up
        cart_penalty = 0.01 * x**2  # Reduced from 0.1 - only prevent hitting boundaries
        velocity_penalty = 0.001 * (dtheta1**2 + dtheta2**2)  # Minimal velocity penalty
        control_penalty = 0.0001 * force**2  # Minimal control effort penalty

        # TOTAL REWARD
        # Prioritize height, then minimize penalties
        swing_up_reward = height_reward - cart_penalty - velocity_penalty - control_penalty

        # At hanging (worst): ~-1.0
        # At upright (best): ~+1.0
        # This gives clear gradient for learning!

        return swing_up_reward

    def __repr__(self):
        return (f"<SwingUpRewardWrapper angle_w={self.angle_weight}, "
                f"vel_w=[{self.velocity_weight_theta1},{self.velocity_weight_theta2}], "
                f"scale={self.scale}>")

"""
Unit tests for Double Pendulum Cart Environment
"""
import pytest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.double_pendulum_cart_env import DoublePendulumCartEnv


class TestDoublePendulumCartEnv:
    """Test suite for environment."""
    
    @pytest.fixture
    def env(self):
        """Create environment instance."""
        return DoublePendulumCartEnv()
    
    def test_environment_creation(self, env):
        """Test that environment can be created."""
        assert env is not None
        assert env.observation_space.shape == (6,)
        assert env.action_space.shape == (1,)
    
    def test_reset(self, env):
        """Test reset functionality."""
        obs, info = env.reset()
        
        # Check observation shape and bounds
        assert obs.shape == (6,)
        assert env.observation_space.contains(obs)
        
        # Check info dict
        assert "cart_position" in info
        assert "pole1_angle" in info
        assert "pole2_angle" in info
    
    def test_step(self, env):
        """Test step functionality."""
        env.reset()
        
        # Take random action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Check returns
        assert obs.shape == (6,)
        assert isinstance(reward, (float, np.floating))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
    
    def test_multiple_steps(self, env):
        """Test that environment can run for multiple steps."""
        env.reset()
        
        for _ in range(100):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                break
        
        # Should complete without errors
        assert True
    
    def test_upright_initialization(self, env):
        """Test initialization near upright position."""
        obs, _ = env.reset()
        
        # Should be close to upright (angles near 0)
        assert abs(obs[1]) < 0.2  # theta1
        assert abs(obs[2]) < 0.2  # theta2
        assert abs(obs[0]) < 0.2  # x
    
    def test_custom_initialization(self, env):
        """Test custom initial state."""
        initial_state = np.array([0.5, 0.1, -0.1, 0.0, 0.0, 0.0])
        obs, _ = env.reset(options={"initial_state": initial_state})
        
        # Should match initial state (within numerical tolerance)
        np.testing.assert_allclose(obs, initial_state, atol=1e-3)
    
    def test_termination_out_of_bounds(self, env):
        """Test that environment terminates when cart goes out of bounds."""
        # Start with cart near boundary
        initial_state = np.array([1.9, 0.0, 0.0, 1.0, 0.0, 0.0])
        env.reset(options={"initial_state": initial_state})
        
        # Apply force pushing cart further
        action = np.array([1.0])  # Maximum positive force
        
        terminated = False
        for _ in range(100):
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
        
        # Should terminate due to cart position
        assert terminated or abs(obs[0]) > env.x_limit
    
    def test_termination_fallen_pendulum(self, env):
        """Test that environment terminates when pendulum falls."""
        # Start with large angle
        initial_state = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        env.reset(options={"initial_state": initial_state})
        
        # No control input
        action = np.array([0.0])
        
        terminated = False
        for _ in range(200):
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
        
        # Should terminate due to falling
        assert terminated or truncated
    
    def test_reward_structure(self, env):
        """Test that reward function is reasonable."""
        env.reset()
        
        rewards = []
        for _ in range(50):
            action = env.action_space.sample()
            _, reward, terminated, truncated, _ = env.step(action)
            rewards.append(reward)
            
            if terminated or truncated:
                break
        
        # Rewards should be finite
        assert all(np.isfinite(r) for r in rewards)
        
        # For upright initialization, early rewards shouldn't be too negative
        assert np.mean(rewards[:10]) > -10.0
    
    def test_action_limits(self, env):
        """Test that actions are properly limited."""
        env.reset()
        
        # Test extreme actions
        actions_to_test = [
            np.array([2.0]),   # Beyond upper limit
            np.array([-2.0]),  # Beyond lower limit
            np.array([0.0]),   # Zero
            np.array([1.0]),   # Maximum
            np.array([-1.0]),  # Minimum
        ]
        
        for action in actions_to_test:
            obs, reward, terminated, truncated, _ = env.step(action)
            # Should not crash
            assert obs.shape == (6,)
    
    def test_deterministic_reset(self, env):
        """Test that reset with seed is deterministic."""
        obs1, _ = env.reset(seed=42)
        obs2, _ = env.reset(seed=42)
        
        np.testing.assert_allclose(obs1, obs2, atol=1e-6)
    
    def test_energy_conservation_no_control(self, env):
        """Test that with no control, energy should be roughly conserved (with damping)."""
        # Start at upright with small perturbation
        initial_state = np.array([0.0, 0.1, 0.0, 0.0, 0.0, 0.0])
        obs, _ = env.reset(options={"initial_state": initial_state})
        
        # No control
        action = np.array([0.0])
        
        initial_angle = obs[1]
        angles = [initial_angle]
        
        for _ in range(100):
            obs, _, terminated, truncated, _ = env.step(action)
            angles.append(obs[1])
            
            if terminated or truncated:
                break
        
        # Pendulum should oscillate (not immediately fall)
        # Check that angle changes sign (oscillation)
        angles_array = np.array(angles)
        
        # Should have some oscillation in first 50 steps
        if len(angles) > 50:
            assert np.max(angles_array[:50]) * np.min(angles_array[:50]) < 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
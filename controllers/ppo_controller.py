"""
PPO Controller for Double Pendulum Cart System.

This controller uses Proximal Policy Optimization (PPO) from stable-baselines3
to learn a nonlinear control policy for stabilizing the double pendulum.

Key features:
- Inherits from RLController base class
- Uses sin/cos angle encoding for observations
- Supports curriculum learning (3°, 10°, 30° levels)
- Compatible with MuJoCo physics simulation
- Provides same interface as LQR/MPC controllers
"""
import numpy as np
import os
from typing import Dict, Any, Optional
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from controllers.base_controller import RLController
from env.double_pendulum_cart_env import DoublePendulumCartEnv
from env.angle_wrapper import AngleObservationWrapper, CurriculumInitializationWrapper


class PPOController(RLController):
    """
    PPO-based controller for double pendulum cart.

    Example usage:
        # Create and train controller
        controller = PPOController(name="PPO_Level1")
        env = create_ppo_env(curriculum_level=1)
        controller.train(env, total_timesteps=100000)

        # Use for control
        state = np.array([0.0, np.pi, np.pi, 0.0, 0.0, 0.0])
        control = controller.compute_control(state)
    """

    def __init__(
        self,
        name: str = "PPO",
        params: Optional[Dict[str, Any]] = None,
        model_path: Optional[str] = None,
    ):
        """
        Initialize PPO controller.

        Args:
            name: Controller name for logging
            params: PPO hyperparameters (uses defaults if None)
            model_path: Path to pre-trained model (if loading)
        """
        # Default PPO hyperparameters (conservative, proven for continuous control)
        default_params = {
            "policy": "MlpPolicy",
            "learning_rate": 3e-4,
            "n_steps": 2048,  # Steps per environment before update
            "batch_size": 64,
            "n_epochs": 10,  # Optimization epochs per update
            "gamma": 0.99,  # Discount factor
            "gae_lambda": 0.95,  # GAE parameter
            "clip_range": 0.2,  # PPO clip parameter
            "clip_range_vf": None,  # Value function clipping (None = no clip)
            "ent_coef": 0.0,  # Entropy coefficient (0 = no entropy bonus)
            "vf_coef": 0.5,  # Value function coefficient
            "max_grad_norm": 0.5,  # Gradient clipping
            "policy_kwargs": {"net_arch": [64, 64]},  # Small network for this task
            "verbose": 1,
        }

        if params is not None:
            default_params.update(params)

        super().__init__(name, default_params)

        # Load model if path provided
        if model_path is not None and os.path.exists(model_path):
            self.load(model_path)
        else:
            self.policy = None
            self.is_trained = False

        # Track training metrics
        self.training_history = {
            "timesteps": [],
            "mean_reward": [],
            "mean_episode_length": [],
        }

    def train(
        self,
        env,
        total_timesteps: int,
        eval_freq: int = 10000,
        save_path: Optional[str] = None,
        callback: Optional[BaseCallback] = None,
    ):
        """
        Train PPO policy.

        Args:
            env: Training environment (should be vectorized)
            total_timesteps: Total training timesteps
            eval_freq: Evaluation frequency (in timesteps)
            save_path: Directory to save checkpoints and best model
            callback: Additional callbacks for training
        """
        print(f"\n{'='*70}")
        print(f"Training PPO Controller: {self.name}")
        print(f"{'='*70}")
        print(f"Total timesteps: {total_timesteps:,}")
        print(f"PPO Hyperparameters:")
        for key, value in self.params.items():
            if key != "policy_kwargs":
                print(f"  {key}: {value}")
        print(f"  Network architecture: {self.params['policy_kwargs']['net_arch']}")
        print(f"{'='*70}\n")

        # Create PPO model if not already created
        if self.policy is None:
            self.policy = PPO(
                policy=self.params["policy"],
                env=env,
                learning_rate=self.params["learning_rate"],
                n_steps=self.params["n_steps"],
                batch_size=self.params["batch_size"],
                n_epochs=self.params["n_epochs"],
                gamma=self.params["gamma"],
                gae_lambda=self.params["gae_lambda"],
                clip_range=self.params["clip_range"],
                clip_range_vf=self.params["clip_range_vf"],
                ent_coef=self.params["ent_coef"],
                vf_coef=self.params["vf_coef"],
                max_grad_norm=self.params["max_grad_norm"],
                policy_kwargs=self.params["policy_kwargs"],
                verbose=self.params["verbose"],
                tensorboard_log=save_path,
            )

        # Setup callbacks
        callbacks = []
        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)

            # Checkpoint callback (save every N steps)
            checkpoint_callback = CheckpointCallback(
                save_freq=max(eval_freq, 10000),
                save_path=os.path.join(save_path, "checkpoints"),
                name_prefix=self.name,
            )
            callbacks.append(checkpoint_callback)

        if callback is not None:
            callbacks.append(callback)

        # Train the model
        self.policy.learn(
            total_timesteps=total_timesteps,
            callback=callbacks if callbacks else None,
            progress_bar=True,
        )

        self.is_trained = True
        print(f"\n✅ Training complete!")

        # Save final model
        if save_path is not None:
            final_model_path = os.path.join(save_path, f"{self.name}_final.zip")
            self.save(final_model_path)
            print(f"Final model saved to: {final_model_path}")

    def compute_control(self, state: np.ndarray) -> np.ndarray:
        """
        Compute control action using trained PPO policy.

        Args:
            state: Current state [x, θ₁, θ₂, ẋ, θ̇₁, θ̇₂] (original 6D state)

        Returns:
            control: Control action (normalized force in [-1, 1])

        Note:
            The state needs to be transformed to sin/cos representation
            internally before passing to the policy network.
        """
        if not self.is_trained or self.policy is None:
            raise RuntimeError("PPO policy not trained or loaded. Call train() or load() first.")

        # Transform state to sin/cos representation (8D)
        x, theta1, theta2, dx, dtheta1, dtheta2 = state

        obs = np.array([
            x,
            np.sin(theta1),
            np.cos(theta1),
            np.sin(theta2),
            np.cos(theta2),
            dx,
            dtheta1,
            dtheta2
        ], dtype=np.float32)

        # Get action from policy (deterministic for control)
        action, _ = self.policy.predict(obs, deterministic=True)

        return action

    def load(self, path: str):
        """
        Load trained PPO policy from disk.

        Args:
            path: Path to saved model (.zip file)
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")

        print(f"Loading PPO model from: {path}")
        self.policy = PPO.load(path)
        self.is_trained = True
        print("✅ Model loaded successfully")

    def save(self, path: str):
        """
        Save trained PPO policy to disk.

        Args:
            path: Path to save model (.zip file)
        """
        if not self.is_trained or self.policy is None:
            raise RuntimeError("No trained policy to save")

        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.policy.save(path)
        print(f"Model saved to: {path}")

    def get_type(self) -> str:
        """Return controller type identifier."""
        return "PPO"

    def get_policy_info(self) -> Dict[str, Any]:
        """
        Get information about the trained policy.

        Returns:
            Dictionary with policy architecture and training info
        """
        if self.policy is None:
            return {"status": "not_initialized"}

        return {
            "status": "trained" if self.is_trained else "initialized",
            "policy_class": str(type(self.policy.policy)),
            "observation_space": str(self.policy.observation_space),
            "action_space": str(self.policy.action_space),
            "n_parameters": sum(p.numel() for p in self.policy.policy.parameters()),
            "device": str(self.policy.device),
        }


def create_ppo_env(
    curriculum_level: int = 1,
    n_envs: int = 1,
    seed: int = 42,
    monitor: bool = True,
) -> Any:
    """
    Create wrapped environment for PPO training.

    Args:
        curriculum_level: Difficulty level (1=±3°, 2=±10°, 3=±30°)
        n_envs: Number of parallel environments
        seed: Random seed
        monitor: Whether to wrap with Monitor for logging

    Returns:
        Vectorized environment ready for PPO training
    """

    def make_env(rank: int):
        """Factory function to create single environment."""

        def _init():
            env = DoublePendulumCartEnv()
            env = CurriculumInitializationWrapper(env, curriculum_level=curriculum_level)
            env = AngleObservationWrapper(env)

            if monitor:
                env = Monitor(env)

            env.reset(seed=seed + rank)
            return env

        return _init

    # Create vectorized environment
    if n_envs == 1:
        # Single environment (simpler, good for debugging)
        env = DummyVecEnv([make_env(0)])
    else:
        # Multiple parallel environments (faster training)
        env = SubprocVecEnv([make_env(i) for i in range(n_envs)])

    return env


# Success rate callback for tracking performance
class SuccessRateCallback(BaseCallback):
    """
    Callback to compute success rate during training.

    Success criterion: Final angle error < 3° (0.052 rad)
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_angle_errors = []
        self.success_threshold = np.deg2rad(3)  # 3 degrees

    def _on_step(self) -> bool:
        # Check if episode ended
        if self.locals.get("dones") is not None:
            dones = self.locals["dones"]
            infos = self.locals["infos"]

            for i, done in enumerate(dones):
                if done and "upright_error" in infos[i]:
                    error = infos[i]["upright_error"]
                    self.episode_angle_errors.append(error)

        return True

    def _on_rollout_end(self) -> None:
        if len(self.episode_angle_errors) > 0:
            successes = [err < self.success_threshold for err in self.episode_angle_errors]
            success_rate = np.mean(successes) * 100

            self.logger.record("rollout/success_rate", success_rate)
            self.logger.record("rollout/mean_angle_error", np.mean(self.episode_angle_errors))
            self.episode_angle_errors = []

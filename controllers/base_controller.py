"""
Base controller class defining common interface for all controllers.
"""
from abc import ABC, abstractmethod
import numpy as np
import time
from typing import Dict, Any, Optional


class BaseController(ABC):
    """
    Abstract base class for all controllers.
    Ensures consistent interface across LQR, MPC, and PPO.
    """
    
    def __init__(self, name: str, params: Optional[Dict[str, Any]] = None):
        """
        Initialize controller.
        
        Args:
            name: Controller name for logging
            params: Dictionary of controller parameters
        """
        self.name = name
        self.params = params or {}
        
        # Performance tracking
        self.computation_times = []
        self.control_history = []
        self.state_history = []
        
    @abstractmethod
    def compute_control(self, state: np.ndarray) -> np.ndarray:
        """
        Compute control action given current state.
        
        Args:
            state: Current state vector [x, θ₁, θ₂, ẋ, θ̇₁, θ̇₂]
            
        Returns:
            control: Control action (force)
        """
        pass
    
    def compute_control_timed(self, state: np.ndarray) -> tuple:
        """
        Compute control and track computation time.
        
        Args:
            state: Current state vector
            
        Returns:
            (control, computation_time): Control action and time taken
        """
        start_time = time.perf_counter()
        control = self.compute_control(state)
        computation_time = time.perf_counter() - start_time
        
        # Log computation time
        self.computation_times.append(computation_time)
        
        return control, computation_time
    
    def log_step(self, state: np.ndarray, control: np.ndarray):
        """Log state and control for later analysis."""
        self.state_history.append(state.copy())
        self.control_history.append(control.copy())
    
    def reset(self):
        """Reset controller state and clear history."""
        self.computation_times = []
        self.control_history = []
        self.state_history = []
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get performance statistics.
        
        Returns:
            Dictionary with computation time stats and control metrics
        """
        if not self.computation_times:
            return {}
        
        comp_times = np.array(self.computation_times)
        controls = np.array(self.control_history) if self.control_history else None
        states = np.array(self.state_history) if self.state_history else None
        
        stats = {
            "controller_name": self.name,
            "computation_time_mean": np.mean(comp_times),
            "computation_time_std": np.std(comp_times),
            "computation_time_max": np.max(comp_times),
            "computation_time_median": np.median(comp_times),
            "total_steps": len(comp_times),
        }
        
        if controls is not None:
            stats.update({
                "control_mean": np.mean(np.abs(controls)),
                "control_max": np.max(np.abs(controls)),
                "control_effort": np.sum(controls**2),
            })
        
        if states is not None:
            # Compute trajectory metrics
            angles = states[:, 1:3]  # θ₁, θ₂
            positions = states[:, 0]  # x
            
            stats.update({
                "max_angle_deviation": np.max(np.abs(angles)),
                "mean_angle_deviation": np.mean(np.abs(angles)),
                "max_position_deviation": np.max(np.abs(positions)),
                "final_angle_error": np.linalg.norm(states[-1, 1:3]),
            })
        
        return stats
    
    def get_params(self) -> Dict[str, Any]:
        """Get controller parameters."""
        return self.params.copy()
    
    def set_params(self, params: Dict[str, Any]):
        """Update controller parameters."""
        self.params.update(params)
    
    @abstractmethod
    def get_type(self) -> str:
        """Return controller type identifier."""
        pass
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name})"


class ModelBasedController(BaseController):
    """
    Base class for model-based controllers (LQR, MPC).
    Provides common functionality for linearization and model parameters.
    """
    
    def __init__(self, name: str, A: np.ndarray, B: np.ndarray, 
                 params: Optional[Dict[str, Any]] = None):
        """
        Initialize model-based controller.
        
        Args:
            name: Controller name
            A: State matrix (6x6)
            B: Input matrix (6x1)
            params: Additional parameters
        """
        super().__init__(name, params)
        self.A = A
        self.B = B
        
        # Store model dimensions
        self.n_states = A.shape[0]
        self.n_inputs = B.shape[1]
        
        # Verify model properties
        self._verify_model()
    
    def _verify_model(self):
        """Verify that system model is valid."""
        assert self.A.shape[0] == self.A.shape[1], "A must be square"
        assert self.B.shape[0] == self.A.shape[0], "B must match A dimensions"
        
        # Check controllability
        from scipy.linalg import matrix_rank
        
        C = self.B
        for i in range(1, self.n_states):
            C = np.hstack((C, np.linalg.matrix_power(self.A, i) @ self.B))
        
        rank = matrix_rank(C)
        if rank < self.n_states:
            print(f"Warning: System may not be fully controllable "
                  f"(rank={rank}, expected={self.n_states})")
    
    def get_model(self) -> tuple:
        """Return system matrices."""
        return self.A.copy(), self.B.copy()


class RLController(BaseController):
    """
    Base class for reinforcement learning controllers (PPO).
    Provides common functionality for policy-based control.
    """
    
    def __init__(self, name: str, params: Optional[Dict[str, Any]] = None):
        """
        Initialize RL controller.
        
        Args:
            name: Controller name
            params: Training and policy parameters
        """
        super().__init__(name, params)
        self.policy = None
        self.is_trained = False
    
    @abstractmethod
    def train(self, env, total_timesteps: int):
        """Train the RL policy."""
        pass
    
    @abstractmethod
    def load(self, path: str):
        """Load trained policy."""
        pass
    
    @abstractmethod
    def save(self, path: str):
        """Save trained policy."""
        pass
    
    def get_type(self) -> str:
        return "RL"

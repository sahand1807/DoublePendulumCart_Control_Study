"""
Linear Quadratic Regulator (LQR) Controller for Double Pendulum Cart

This controller stabilizes the double pendulum at the upright equilibrium
using optimal state feedback based on the linearized system dynamics.

Equilibrium: x_eq = [0, π, π, 0, 0, 0] (cart centered, both pendulums upright)
Control law: u = -K(x - x_eq)
"""

import numpy as np
from scipy.linalg import solve_continuous_are
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from controllers.base_controller import ModelBasedController


class LQRController(ModelBasedController):
    """
    LQR controller for double pendulum cart system.
    
    Computes optimal feedback gain K by solving the Algebraic Riccati Equation.
    """
    
    def __init__(self, A, B, Q=None, R=None, name="LQR"):
        """
        Initialize LQR controller.
        
        Args:
            A: State matrix (6x6) from linearization
            B: Input matrix (6x1) from linearization
            Q: State weighting matrix (6x6), default: diag([10, 100, 100, 1, 10, 10])
            R: Control weighting matrix (1x1), default: 1.0
            name: Controller name for logging
        """
        super().__init__(name, A, B, params={})
        
        # Default Q matrix: balanced weighting
        if Q is None:
            Q = np.diag([
                1.0,    # x: cart position (low priority)
                100.0,   # θ₁: first pendulum angle  
                100.0,   # θ₂: second pendulum angle
                0.1,    # ẋ: cart velocity (very low)
                10.0,    # θ̇₁: first pendulum angular velocity
                10.0     # θ̇₂: second pendulum angular velocity
            ])
        
        # Default R matrix: penalize control effort more
        if R is None:
            R = np.array([[1.0]])  # Larger R = less aggressive control
        
        self.Q = Q
        self.R = R
        
        # Store parameters
        self.params = {
            'Q': Q.copy(),
            'R': R.copy(),
        }
        
        # Solve Algebraic Riccati Equation
        print(f"Solving Algebraic Riccati Equation for {name}...")
        self.P = solve_continuous_are(A, B, Q, R)
        
        # Compute optimal gain
        self.K = np.linalg.solve(R, B.T @ self.P)
        
        # Store for later use
        self.params['K'] = self.K.copy()
        self.params['P'] = self.P.copy()
        
        print(f"✓ LQR gain computed: K shape = {self.K.shape}")
        print(f"  Gain values: {self.K.flatten()}")
        
        # Verify closed-loop stability
        A_cl = A - B @ self.K
        eigenvalues = np.linalg.eigvals(A_cl)
        
        if np.all(np.real(eigenvalues) < 0):
            print(f"✓ Closed-loop system is stable")
            print(f"  Eigenvalues: {eigenvalues}")
        else:
            print(f"⚠ Warning: Closed-loop system may be unstable!")
            print(f"  Eigenvalues: {eigenvalues}")
        
        # Equilibrium point (upright)
        self.x_eq = np.array([0.0, np.pi, np.pi, 0.0, 0.0, 0.0])
    
    def compute_control(self, state):
        """
        Compute optimal control using LQR feedback.
        
        Args:
            state: Current state [x, θ₁, θ₂, ẋ, θ̇₁, θ̇₂]
                   Angles are absolute (θ = π is upright)
        
        Returns:
            u: Control force (scalar)
        """
        # Compute state error relative to upright equilibrium
        state_error = state - self.x_eq
        
        # Wrap angle errors to [-π, π]
        state_error[1] = self._wrap_angle(state_error[1])  # θ₁ error
        state_error[2] = self._wrap_angle(state_error[2])  # θ₂ error
        
        # Optimal control: u = -K * (x - x_eq)
        u = -self.K @ state_error
        
        # Return scalar control
        return u.flatten()[0]
    
    def _wrap_angle(self, angle):
        """
        Wrap angle to [-π, π].
        
        Args:
            angle: Angle in radians
        
        Returns:
            Wrapped angle in [-π, π]
        """
        return np.arctan2(np.sin(angle), np.cos(angle))
    
    def set_reference(self, x_ref):
        """
        Set reference equilibrium point.
        
        Args:
            x_ref: Reference state [x, θ₁, θ₂, ẋ, θ̇₁, θ̇₂]
        """
        self.x_eq = x_ref.copy()
    
    def get_gain(self):
        """Return the LQR gain matrix K."""
        return self.K.copy()
    
    def get_riccati_solution(self):
        """Return the Riccati solution P."""
        return self.P.copy()
    
    def get_type(self):
        """Return controller type."""
        return "LQR"
    
    def get_info(self):
        """
        Get detailed controller information.
        
        Returns:
            Dictionary with controller details
        """
        A_cl = self.A - self.B @ self.K
        eigenvalues = np.linalg.eigvals(A_cl)
        
        return {
            'type': 'LQR',
            'gain_K': self.K,
            'riccati_P': self.P,
            'Q_matrix': self.Q,
            'R_matrix': self.R,
            'equilibrium': self.x_eq,
            'closed_loop_eigenvalues': eigenvalues,
            'is_stable': np.all(np.real(eigenvalues) < 0),
        }


def create_lqr_controller(Q=None, R=None):
    """
    Factory function to create LQR controller with linearized system.
    
    Args:
        Q: State weighting matrix (6x6)
        R: Control weighting matrix (1x1)
    
    Returns:
        LQRController instance
    """
    # Import linearization utilities
    import sys
    import os
    
    # Add src directory to path
    src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    
    from linearization import linearize_system
    from system import params_default
    
    # Equilibrium point (upright) - using ABSOLUTE angles as in derivation
    q_eq = [0.0, np.pi, np.pi]  # x, θ₁_abs, θ₂_abs
    dq_eq = [0.0, 0.0, 0.0]     # ẋ, θ̇₁, θ̇₂
    u_eq = 0.0
    
    # Compute linearization
    print("Computing linearization around upright equilibrium...")
    A, B = linearize_system(q_eq, dq_eq, u_eq, params_default)
    print(f"✓ Linearization complete: A shape = {A.shape}, B shape = {B.shape}")
    
    # Create controller
    controller = LQRController(A, B, Q, R)
    
    return controller


if __name__ == "__main__":
    """Test LQR controller creation."""
    print("=" * 60)
    print("Testing LQR Controller")
    print("=" * 60)
    
    # Create controller with default Q, R
    controller = create_lqr_controller()
    
    print("\n" + "=" * 60)
    print("Controller Information:")
    print("=" * 60)
    
    info = controller.get_info()
    print(f"\nController type: {info['type']}")
    print(f"Stable: {info['is_stable']}")
    print(f"\nGain K:\n{info['gain_K']}")
    print(f"\nQ matrix:\n{np.diag(info['Q_matrix'])}")
    print(f"R matrix: {info['R_matrix'][0,0]}")
    
    print("\n" + "=" * 60)
    print("Testing control computation...")
    print("=" * 60)
    
    # Test with small perturbation from upright
    test_state = np.array([0.0, np.pi + 0.1, np.pi + 0.05, 0.0, 0.0, 0.0])
    control = controller.compute_control(test_state)
    
    print(f"\nTest state: {test_state}")
    print(f"Control output: {control:.4f} N")
    
    print("\n" + "=" * 60)
    print("✓ LQR Controller test complete!")
    print("=" * 60)
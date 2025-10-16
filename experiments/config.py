"""
Configuration management for experiments
"""
import yaml
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional
import numpy as np


@dataclass
class SystemParams:
    """Physical parameters of the double pendulum cart system."""
    M_cart: float = 1.0      # Cart mass (kg)
    m1: float = 0.3          # First pendulum mass (kg)
    m2: float = 0.2          # Second pendulum mass (kg)
    l1: float = 0.5          # First pendulum length (m)
    l2: float = 0.4          # Second pendulum length (m)
    lc1: float = 0.25        # First pendulum COM distance (m)
    lc2: float = 0.2         # Second pendulum COM distance (m)
    I1: float = 0.01         # First pendulum inertia (kg⋅m²)
    I2: float = 0.008        # Second pendulum inertia (kg⋅m²)
    g: float = 9.81          # Gravity (m/s²)
    
    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class EnvironmentConfig:
    """Environment configuration."""
    max_episode_steps: int = 1000
    frame_skip: int = 5
    max_force: float = 20.0
    x_limit: float = 2.0
    render_mode: Optional[str] = None
    

@dataclass
class LQRConfig:
    """LQR controller configuration."""
    # State weighting matrix Q (diagonal)
    Q_diag
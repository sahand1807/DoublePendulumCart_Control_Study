# Double Pendulum on Cart Control Study

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![MuJoCo](https://img.shields.io/badge/MuJoCo-3.0+-green.svg)](https://mujoco.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A comprehensive study of control strategies for stabilizing a double pendulum on a cart system at the upright equilibrium.


## 🎯 Objective

Design, implement, and compare three control strategies for balancing a double-pendulum-on-cart system at the unstable upright equilibrium:

1. **LQR** (Linear Quadratic Regulator)
2. **MPC** (Model Predictive Control)
3. **PPO** (Proximal Policy Optimization)

The system is simulated using a high-fidelity **MuJoCo** physics engine through a custom **Gymnasium** environment with accurately modeled dynamics derived from Lagrangian mechanics.

---

## 🔬 System Description

### Physical Parameters

| Component | Parameter | Value |
|-----------|-----------|-------|
| **Cart** | Mass (M) | 1.0 kg |
| | Force limit | ±20 N |
| **Link 1** | Mass (m₁) | 0.3 kg |
| | Length (l₁) | 0.5 m |
| | COM distance (lc₁) | 0.25 m |
| | Inertia (I₁) | 0.01 kg⋅m² |
| **Link 2** | Mass (m₂) | 0.2 kg |
| | Length (l₂) | 0.4 m |
| | COM distance (lc₂) | 0.2 m |
| | Inertia (I₂) | 0.008 kg⋅m² |

### State Space

**State vector**: `[x, θ₁, θ₂, ẋ, θ̇₁, θ̇₂]`
- `x`: Cart position (m)
- `θ₁, θ₂`: Absolute angles from vertical (rad)
  - θ = 0: hanging down
  - θ = π: upright (target)
- `ẋ, θ̇₁, θ̇₂`: Velocities

**Control**: Horizontal force on cart (continuous)

**Simulation**: 100 Hz (dt = 0.01s), RK4 integrator

---

## 🚀 Quick Start

### Installation
```bash
# Clone repository
git clone https://github.com/sahand1807/DoublePendulumCart_Control_Study.git
cd DoublePendulumCart_Control_Study

# Create virtual environment
python -m venv venv
source venv/bin/activate 

# Install dependencies
pip install -r requirements.txt

```
---
### Run LQR Controller
```bash
# Run LQR experiments with visualization
python experiments/run_lqr.py

# Real-time 3D viewer with LQR control
python experiments/run_lqr_3D.py

# Test environment
python test_visualization.py
```
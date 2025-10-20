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

## 🤖 Controllers Implemented

### 1. LQR (Linear Quadratic Regulator)
**Status:** ✅ Complete

Classic model-based optimal control approach using linearization around upright equilibrium.

**Features:**
- Analytical solution via Continuous Algebraic Riccati Equation (CARE)
- Fast computation (~0.05 ms per step)
- Guaranteed stability within linear regime
- Region of attraction: ~32.5% of ±30° test grid

**Usage:**
```bash
# Run LQR experiments with visualization
python experiments/run_lqr.py

# Real-time 3D viewer with LQR control
python experiments/run_lqr_3D.py
```

**Documentation:** See [LQR Study Report](docs/LQR_Study_Report.md)

---

### 2. PPO (Proximal Policy Optimization)
**Status:** ✅ Complete

Model-free deep reinforcement learning approach with curriculum learning.

**Features:**
- 3-level curriculum: ±3° → ±6° → ±10° perturbations
- Transfer learning across difficulty levels
- Neural network policy: [64, 64] architecture
- Sin/cos angle encoding for continuous learning
- Hardware acceleration (Apple M2 GPU / CUDA support)

**Training:**
```bash
# Level 1: Train from scratch (±3° perturbations)
python experiments/train_ppo_level1.py --timesteps 500000

# Level 2: Transfer learning (±6° perturbations)
python experiments/train_ppo_level2.py \
    --transfer-from results/ppo_level1/best_model/best_model.zip \
    --timesteps 200000

# Level 3: Transfer learning (±10° perturbations)
python experiments/train_ppo_level3.py \
    --transfer-from results/ppo_level2/best_model/best_model.zip \
    --timesteps 500000
```

**Evaluation & Visualization:**
```bash
# Evaluate trained models
python experiments/evaluate_ppo_level1.py
python experiments/evaluate_ppo_level2.py
python experiments/evaluate_ppo_level3.py

# Generate animations
python experiments/render_ppo_level1.py
python experiments/render_ppo_level2.py
python experiments/render_ppo_level3.py

# Plot learning curves
python utils/plot_learning_curves.py --log-dir results/ppo_level1
```

**Performance:**
- Level 1: 100% success rate, 1.2s settling time
- Level 2: 100% success rate, 2.3s settling time
- Level 3: 95% success rate, 3.6s settling time
- Total training time: ~108 minutes (M2 GPU)

**Documentation:** See [PPO Study Report](docs/PPO_Study_Report.md)

---

### 3. MPC (Model Predictive Control)
**Status:** 🚧 Planned

Nonlinear optimal control with constraint handling via receding horizon optimization.

---

## 📊 Results Summary

| Controller | Success Rate | Settling Time | Cart Drift | Region of Attraction |
|------------|--------------|---------------|------------|---------------------|
| **LQR** | 100% (within ROA) | 0.41s | 0.15m | ~32.5% (±30° grid) |
| **PPO Level 1** | 100% | 1.2s | 0.12m | ±3° (trained) |
| **PPO Level 2** | 100% | 2.3s | 0.24m | ±6° (trained) |
| **PPO Level 3** | 95% | 3.6s | 0.42m | ±10° (trained) |
| **MPC** | TBD | TBD | TBD | TBD |

**Key Findings:**
- **LQR:** Fastest response but limited to small perturbations
- **PPO:** Wider capability range, handles 2× larger perturbations than LQR
- **Hybrid Approach Recommended:** Use PPO for large deviations, switch to LQR for fast settling

---

## 📁 Project Structure

```
DoublePendulumCart_Control_Study/
├── env/                          # Gymnasium environment
│   ├── double_pendulum_cart_env.py
│   ├── angle_wrapper.py          # Sin/cos encoding + curriculum
│   └── double_pendulum_cart.xml  # MuJoCo model
├── controllers/                  # Controller implementations
│   ├── lqr_controller.py
│   ├── ppo_controller.py
│   └── base_controller.py
├── experiments/                  # Training & evaluation scripts
│   ├── train_ppo_level*.py
│   ├── evaluate_ppo_level*.py
│   ├── render_ppo_level*.py
│   ├── run_lqr.py
│   └── run_lqr_3D.py
├── utils/                        # Visualization utilities
│   └── plot_learning_curves.py
├── docs/                         # Documentation & reports
│   ├── PPO_Study_Report.md
│   ├── LQR_Study_Report.md
│   └── theory.md
├── results/                      # Saved models & plots
│   ├── ppo_level1/
│   ├── ppo_level2/
│   └── ppo_level3/
├── assets/                       # Figures for documentation
│   ├── ppo/
│   └── lqr/
└── requirements.txt
```

---

## 🔧 Testing

```bash
# Test environment step function
python tests/test_env_step.py

# Test visualization
python test_visualization.py
```
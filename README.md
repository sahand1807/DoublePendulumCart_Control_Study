# Double Pendulum on Cart Control Study

## Objective
Design, implement, and compare three control strategies for a double-pendulum-on-cart system:
1. LQR (Linear Quadratic Regulator)
2. MPC (Model Predictive Control)
3. PPO (Proximal Policy Optimization)

In this study, we will investigate the problem of controlling a double pendulum on a cart to stay upright. The dynamics will be simulated via a custom Gymnasium environment with MuJoCo simulator.

## Project Structure
- `env/`: Gym environment with continuous dynamics.
- `controllers/`: LQR, MPC, PPO implementations.
- `experiments/`: Experiment scripts and YAML configuration.
- `analysis/`: Visualization and comparison notebooks.
- `docs/`: Theory, derivations, and findings.
- `tests/`: Unit tests for dynamics and controllers.
- `results/`: Experiment logs, plots, and saved models.
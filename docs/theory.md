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


## Theoretical Foundation
Here are the key facts about the dynamics of the system and the control strategies that will be investigated in this project:
- Dynamics: 
    - Cart + Double Pendulum System
    - 6 states: Linear position and velocity of the cart and angulr position and velocity of the pendulums.
    - 1 actuator (cart force). 

- Controls: 
    -  LQR: Linearization around upright equilibrium.  
    - MPC: Nonlinear MPC formulation via CasADi using receding horizon.  
    - PPO: Model-free RL agent trained in MuJoCo environment, with reward shaping to stabilize in the upright position.  

---    
In the follwoing sections we will investigate the dynamics of the system in detail, overview the set-up for the simulation environment and will study the theoretical foundation of the various control and reinforcement learning strategies that are implemented in this project.

## 1. System Definition
The double pendulum on a cart consists of a cart of mass M that moves horizontally, 
with two pendulums (m₁, m₂) attached via rotary joints. The control input u is the horizontal force on the cart.

**State Vector**
State vector consists of 6 variables as follows:
$x = [x_{cart}, θ₁, θ₂, ẋ_{cart}, θ̇₁, θ̇₂]ᵀ$

**Control Input**
The horizontal force applied on the cart:
$u = [F]$

**Objective**
Stabilize both pendulums in the upright position and track reference trajectories when needed.


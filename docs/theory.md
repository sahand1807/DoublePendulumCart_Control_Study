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
The double pendulum on a cart consists of:

- A cart of mass \( M \) moving horizontally on a frictionless track.  
- Two pendulum links:
  - Link 1: mass \( m_1 \), length \( l_1 \), center of mass at \( l_{c1} \) from the pivot.
  - Link 2: mass \( m_2 \), length \( l_2 \), center of mass at \( l_{c2} \) from its joint with link 1.
- Control input: a horizontal force \( u \) applied to the cart.

The generalized coordinates are:

$$
q =
\begin{bmatrix}
x \\
\theta_1 \\
\theta_2
\end{bmatrix}
$$

where:
- \( x \): horizontal position of the cart  
- \( \theta_1 \): angle of the first pendulum from the vertical (counterclockwise positive)  
- \( \theta_2 \): angle of the second pendulum from the vertical (counterclockwise positive)

---

## 2. Lagrangian Derivation

### 2.1 Kinetic Energy

Define the generalized coordinates and velocities:
$$
q = \begin{bmatrix} x \\ \theta_1 \\ \theta_2 \end{bmatrix}, \qquad
\dot q = \begin{bmatrix} \dot x \\ \dot\theta_1 \\ \dot\theta_2 \end{bmatrix}.
$$

Assume:
- cart mass: \(M\),
- link 1 mass and COM distance: \(m_1,\, l_{c1}\),
- link 2 mass and COM distance: \(m_2,\, l_{c2}\),
- distance from cart pivot to link-1 joint: \(l_1\),
- rotational inertias about each link's COM: \(I_1, I_2\),

Positions of centers of mass (using absolute angles from vertical):
- COM of link 1:
  $$
  \begin{aligned}
  x_{1} &= x + l_{c1}\sin\theta_1,\\
  y_{1} &= -\,l_{c1}\cos\theta_1,
  \end{aligned}
  $$
- COM of link 2:
  $$
  \begin{aligned}
  x_{2} &= x + l_1\sin\theta_1 + l_{c2}\sin\theta_2,\\
  y_{2} &= -\,l_1\cos\theta_1 - l_{c2}\cos\theta_2.
  \end{aligned}
  $$

Velocities (time derivatives):
- COM 1 velocity components:
  $$
  \begin{aligned}
  \dot x_{1} &= \dot x + l_{c1}\cos\theta_1\,\dot\theta_1,\\
  \dot y_{1} &= l_{c1}\sin\theta_1\,\dot\theta_1,
  \end{aligned}
  $$
  so
  $$
  v_1^2 = \dot x_{1}^2 + \dot y_{1}^2
  = \dot x^2 + 2\dot x\,l_{c1}\cos\theta_1\,\dot\theta_1 + l_{c1}^2\dot\theta_1^2.
  $$

- COM 2 velocity components:
  $$
  \begin{aligned}
  \dot x_{2} &= \dot x + l_1\cos\theta_1\,\dot\theta_1 + l_{c2}\cos\theta_2\,\dot\theta_2,\\
  \dot y_{2} &= l_1\sin\theta_1\,\dot\theta_1 + l_{c2}\sin\theta_2\,\dot\theta_2,
  \end{aligned}
  $$
  hence
  $$
  \begin{aligned}
  v_2^2 &= \dot x_{2}^2 + \dot y_{2}^2 \\
  &= \dot x^2 + 2\dot x\bigl(l_1\cos\theta_1\,\dot\theta_1 + l_{c2}\cos\theta_2\,\dot\theta_2\bigr) \\
  &\quad + \bigl(l_1\cos\theta_1\,\dot\theta_1 + l_{c2}\cos\theta_2\,\dot\theta_2\bigr)^2 + \bigl(l_1\sin\theta_1\,\dot\theta_1 + l_{c2}\sin\theta_2\,\dot\theta_2\bigr)^2 \\
  &= \dot x^2 + 2\dot x\bigl(l_1\cos\theta_1\,\dot\theta_1 + l_{c2}\cos\theta_2\,\dot\theta_2\bigr) \\
  &\quad + l_1^2\dot\theta_1^2 + l_{c2}^2\dot\theta_2^2 + 2 l_1 l_{c2}\cos(\theta_1-\theta_2)\,\dot\theta_1\dot\theta_2.
  \end{aligned}
  $$

Now we can write the kinetic energy components:

- Cart translational kinetic energy:
  $$
  T_{\text{cart}} = \tfrac{1}{2} M \dot x^2.
  $$

- Link 1 translational + rotational kinetic energy:
  $$
  T_1 = \tfrac{1}{2} m_1 v_1^2 + \tfrac{1}{2} I_1 \dot\theta_1^2
      = \tfrac{1}{2} m_1\bigl(\dot x^2 + 2\dot x\,l_{c1}\cos\theta_1\,\dot\theta_1 + l_{c1}^2\dot\theta_1^2\bigr)
        + \tfrac{1}{2} I_1 \dot\theta_1^2.
  $$

- Link 2 translational + rotational kinetic energy:
  $$
  \begin{aligned}
  T_2 &= \tfrac{1}{2} m_2 v_2^2 + \tfrac{1}{2} I_2 \dot\theta_2^2 \\
      &= \tfrac{1}{2} m_2\Bigl(
           \dot x^2
           + 2\dot x\bigl(l_1\cos\theta_1\,\dot\theta_1 + l_{c2}\cos\theta_2\,\dot\theta_2\bigr) \\
      &\qquad\qquad + l_1^2\dot\theta_1^2 + l_{c2}^2\dot\theta_2^2
           + 2 l_1 l_{c2}\cos(\theta_1-\theta_2)\,\dot\theta_1\dot\theta_2
        \Bigr)
        + \tfrac{1}{2} I_2 \dot\theta_2^2.
  \end{aligned}
  $$

Finally, the **total kinetic energy** is the sum:
$$
\boxed{
\begin{aligned}
T &= T_{\text{cart}} + T_1 + T_2 \\
  &= \tfrac{1}{2} M \dot x^2 \\
  &\quad + \tfrac{1}{2} m_1\bigl(\dot x^2 + 2\dot x\,l_{c1}\cos\theta_1\,\dot\theta_1 + l_{c1}^2\dot\theta_1^2\bigr)
        + \tfrac{1}{2} I_1 \dot\theta_1^2 \\
  &\quad + \tfrac{1}{2} m_2\Bigl(
           \dot x^2
           + 2\dot x\bigl(l_1\cos\theta_1\,\dot\theta_1 + l_{c2}\cos\theta_2\,\dot\theta_2\bigr) \\
  &\qquad\qquad\qquad\qquad\quad
           + l_1^2\dot\theta_1^2 + l_{c2}^2\dot\theta_2^2
           + 2 l_1 l_{c2}\cos(\theta_1-\theta_2)\,\dot\theta_1\dot\theta_2
        \Bigr)
        + \tfrac{1}{2} I_2 \dot\theta_2^2.
\end{aligned}
}
$$

**Notes & remarks**
- The terms proportional to \(\dot x^2\) appear in every translational kinetic energy and can be collected together when forming the mass/inertia matrix \(M(q)\).
- Cross terms of the form \(\dot x\,\dot\theta_i\) represent coupling between cart translation and pendulum rotations (these contribute to the off-diagonal inertia terms).
- The term \(2 l_1 l_{c2}\cos(\theta_1-\theta_2)\,\dot\theta_1\dot\theta_2\) is the coupling between link 1 and link 2 angular velocities (depends on relative angle).


### 2.2 Potential Energy
The potential energy of the system can be derived as follows:

$$
V = m_1 g l_{c1} \cos \theta_1 + m_2 g \left( l_1 \cos \theta_1 + l_{c2} \cos \theta_2 \right)
$$


### 2.3 Lagrangian
The Lagrangian of the system is defined as:
$$
\mathcal{L} = T - V
$$

---

### 2.4 Equations of Motion

From the Euler–Lagrange equation:

$$
\frac{d}{dt}\left( \frac{\partial \mathcal{L}}{\partial \dot{q}} \right) - \frac{\partial \mathcal{L}}{\partial q} = Q
$$

where \( Q = [u,\, 0,\, 0]^T \) represents the generalized forces (the input force acts only on the cart).

The resulting system can be expressed in the standard form as:

$$
M(q) \ddot{q} + C(q, \dot{q}) \dot{q} + G(q) = B u
$$

where:
- \( M(q) \): inertia matrix  
- \( C(q, \dot{q}) \): Coriolis and centrifugal terms  
- \( G(q) \): gravity vector  
- \( B = [1,\, 0,\, 0]^T \)

---

## 3. Linearization

We linearize the nonlinear system around an equilibrium point \( (q_{eq}, \dot{q}_{eq}, u_{eq}) \) to obtain a linear approximation:

$$
\dot{x} = A x + B u
$$

where the state vector is:

$$
x = 
\begin{bmatrix}
x \\ \theta_1 \\ \theta_2 \\ \dot{x} \\ \dot{\theta}_1 \\ \dot{\theta}_2
\end{bmatrix}
$$

and \(u\) is the force applied on the cart. The linearization matrices are defined as:

$$
A = \left. \frac{\partial f(x,u)}{\partial x} \right|_{x_{eq},u_{eq}}, \quad
B = \left. \frac{\partial f(x,u)}{\partial u} \right|_{x_{eq},u_{eq}}
$$

where \( f(x,u) \) is obtained from the numeric dynamics:

$$
\ddot{q} = f(q, \dot{q}, u)
$$

---

### 4. Controllability

We verify controllability by constructing the controllability matrix:

$$
\mathcal{C} = [B, AB, A^2 B, \dots, A^{n-1} B]
$$

The system is controllable if \( \text{rank}(\mathcal{C}) = n \), where \( n \) is the dimension of the state vector.



### 4.1 Linearized State-Space Matrices and Controllability
Running `linearization_controllability_check.py` we get:

$$
A =
\begin{bmatrix}
0 & 0 & 0 & 1 & 0 & 0 \\
0 & 0 & 0 & 0 & 1 & 0 \\
0 & 0 & 0 & 0 & 0 & 1 \\
0 & -1.5937 & -0.8145 & 0 & 0 & 0 \\
0 & 29.7487 & 2.1249 & 0 & 0 & 0 \\
0 & 3.9842 & 26.5614 & 0 & 0 & 0
\end{bmatrix},
\quad
B =
\begin{bmatrix}
0 \\ 0 \\ 0 \\ -0.8303 \\ 2.1661 \\ 2.0758
\end{bmatrix}
$$

### Observations

1. The system has **6 states**: 3 positions and 3 velocities.  
2. The top-right **identity block in A** represents the relationship $\dot{q} = \text{velocity}$.  
3. The bottom-left block in A captures the sensitivity of accelerations to positions (∂ddq/∂q), while the bottom-right block (∂ddq/∂dq) is zero because there are no velocity-dependent forces (no damping).  
4. The B matrix shows how the input force affects accelerations. Top rows are zero because positions do not directly respond to the input.  
5. The controllability matrix has **full rank (6)**, confirming that all states can be controlled with a single input force applied to the cart.  
6. This linearized model provides the foundation for designing **LQR and MPC controllers**, and serves as a comparison benchmark for reinforcement learning approaches.







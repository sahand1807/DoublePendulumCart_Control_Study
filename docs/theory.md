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


## 5. Linear Quadratic Regulator (LQR)

### 5.1 Problem Formulation

For the linearized system around the upright equilibrium:

$$
\dot{x} = A x + B u
$$

where the state vector is:

$$
x = \begin{bmatrix}
x \\ \theta_1 \\ \theta_2 \\ \dot{x} \\ \dot{\theta}_1 \\ \dot{\theta}_2
\end{bmatrix}
$$

The LQR problem seeks to find the control input \( u(t) \) that minimizes the infinite-horizon quadratic cost function:

$$
J = \int_0^\infty \left( x^T Q x + u^T R u \right) dt
$$

where:
- \( Q \in \mathbb{R}^{6 \times 6} \) is a positive semi-definite state weighting matrix
- \( R \in \mathbb{R}^{1 \times 1} \) is a positive definite control weighting matrix

The matrix \( Q \) penalizes deviations of the state from the equilibrium, while \( R \) penalizes control effort (large forces on the cart).

---

### 5.2 Optimal Control Law

The optimal control law that minimizes the cost function \( J \) is given by the linear state feedback:

$$
u^*(t) = -K x(t)
$$

where \( K \in \mathbb{R}^{1 \times 6} \) is the optimal feedback gain matrix computed as:

$$
K = R^{-1} B^T P
$$

The matrix \( P \in \mathbb{R}^{6 \times 6} \) is the unique positive definite solution to the **Continuous-time Algebraic Riccati Equation (ARE)**:

$$
A^T P + P A - P B R^{-1} B^T P + Q = 0
$$

---

### 5.3 Closed-Loop Stability

With the optimal control \( u = -Kx \), the closed-loop system becomes:

$$
\dot{x} = (A - BK) x
$$

The LQR solution guarantees that:
1. The matrix \( (A - BK) \) is **Hurwitz** (all eigenvalues have negative real parts)
2. The closed-loop system is **asymptotically stable**
3. All states converge to the equilibrium: \( x(t) \to 0 \) as \( t \to \infty \)

The Lyapunov function:

$$
V(x) = x^T P x
$$

satisfies \( \dot{V} < 0 \) for all \( x \neq 0 \), proving stability.

---

### 5.4 Design Choices: Q and R Matrices

The weighting matrices \( Q \) and \( R \) must be chosen to reflect the control objectives and physical constraints.

#### State Weighting Matrix Q

For our double pendulum cart system, we choose \( Q \) as a diagonal matrix:

$$
Q = \text{diag}(q_x, q_{\theta_1}, q_{\theta_2}, q_{\dot{x}}, q_{\dot{\theta}_1}, q_{\dot{\theta}_2})
$$

**Design rationale**:
- **Large weights on angles** \( q_{\theta_1} \) and \( q_{\theta_2} \): The primary objective is to keep both pendulums upright. These should be the largest weights.
- **Moderate weight on cart position** \( q_x \): We want the cart to stay near the center, but this is secondary to balancing.
- **Small weights on velocities**: Penalize rapid motions, but less critical than position errors.

**Typical values**:
$$
Q = \text{diag}(10, 100, 100, 1, 10, 10)
$$

#### Control Weighting Matrix R

The scalar \( R \) penalizes control effort:

$$
R = r > 0
$$

**Design rationale**:
- **Larger R**: More conservative control, smaller forces, slower response
- **Smaller R**: Aggressive control, larger forces, faster response
- In this study we consider balancing performance with an imaginary actuator limit (\( |u| \leq 20 \) N)

**Typical value**:
$$
R = 1
$$

This choice allows sufficient control authority while avoiding excessive force.

---

### 5.5 Implementation Procedure

The LQR controller is implemented using the following steps:

1. **Linearize** the nonlinear dynamics around the upright equilibrium to obtain \( A \) and \( B \)
2. **Verify controllability** by checking that \( \text{rank}(\mathcal{C}) = 6 \)
3. **Choose** weighting matrices \( Q \) and \( R \) based on control objectives
4. **Solve the ARE** using numerical methods (e.g., `scipy.linalg.solve_continuous_are`)
5. **Compute the gain** \( K = R^{-1} B^T P \)
6. **Apply control** \( u = -K(x - x_{\text{eq}}) \) where \( x_{\text{eq}} = [0, \pi, \pi, 0, 0, 0]^T \)

**Note on angle wrapping**: Since the equilibrium angles are \( \theta_{eq} = \pi \), the control error must account for angle wrapping:

$$
e_{\theta} = \text{wrap}(\theta - \pi)
$$

where \( \text{wrap}(\theta) \) maps angles to \( [-\pi, \pi] \).

Using the default weighting matrices, the LQR controller produces the following optimal gain:

$$
K = \begin{bmatrix}
-3.16 & -16.87 & 2.84 & -4.42 & -0.51 & -3.74
\end{bmatrix}
$$

The closed-loop eigenvalues are:

$$
\lambda = \{-5.34 \pm 3.80i,\, -0.88 \pm 0.87i,\, -0.05 \pm 5.00i\}
$$

All eigenvalues have negative real parts, confirming asymptotic stability of the closed-loop system.

---

### 5.6 Limitations and Region of Attraction

The LQR controller is designed based on the **linearized** system, which is only valid near the equilibrium point. 

**Key limitations**:

1. **Local stability only**: The controller is guaranteed to work only for **small deviations** from upright. For large initial angles, the linearization is invalid and performance degrades.

2. **No constraint handling**: LQR does not explicitly handle state or input constraints. If the optimal control exceeds actuator limits, it must be saturated, which can degrade performance.

3. **Model-based**: Requires accurate knowledge of system parameters. Parameter uncertainty or modeling errors can affect performance.

4. **Region of attraction**: The domain in which LQR successfully stabilizes the system must be determined experimentally.

---

### 5.7 Performance Metrics

To evaluate the LQR controller, we use the following metrics:

1. **Settling time** \( t_s \): Time for \( \|\theta - \pi\| < 0.05 \) rad (≈3°)
2. **Maximum overshoot**: Peak deviation during transient response
3. **Control effort**: \( E = \int_0^T u^2 dt \)
4. **Computation time**: Time per control step (should be < 1 ms for real-time feasibility)
5. **Success rate**: Percentage of initial conditions that converge to equilibrium







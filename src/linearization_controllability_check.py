import numpy as np
from system import ddq_numeric, params_default
from linearization import linearize_system, is_controllable

# Equilibrium (upright pendulums)
q_eq = [0.0, 0.0, 0.0]         # x, theta1, theta2
dq_eq = [0.0, 0.0, 0.0]        # dx, dtheta1, dtheta2
u_eq = 0.0                      # no force

# Compute linearization
A, B = linearize_system(q_eq, dq_eq, u_eq, params_default)

# Check controllability
controllable, rank = is_controllable(A, B)

print("A matrix:\n", A)
print("B matrix:\n", B)
print("Controllable:", controllable, "Rank:", rank)

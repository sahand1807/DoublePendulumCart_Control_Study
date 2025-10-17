# src/linearization.py
import numpy as np
import os
import sys

# Get the directory containing this file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add it to path if not already there
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Now import system module (same directory)
from system import ddq_numeric

def linearize_system(q_eq, dq_eq, u_eq, params=None, eps=1e-5):
    n = len(q_eq) # 3
    m = len(dq_eq) # 3
    A_ddq_q = np.zeros((n,n))
    A_ddq_dq = np.zeros((n,n))
    B_ddq_u = np.zeros((n,1))
    # ddq = f(q,dq,u)
    for i in range(n):
        q_plus = q_eq.copy(); q_plus[i] += eps
        q_minus = q_eq.copy(); q_minus[i] -= eps
        f_plus = ddq_numeric(q_plus, dq_eq, u_eq, params)
        f_minus = ddq_numeric(q_minus, dq_eq, u_eq, params)
        A_ddq_q[:,i] = (f_plus - f_minus)/(2*eps)
        dq_plus = dq_eq.copy(); dq_plus[i] += eps
        dq_minus = dq_eq.copy(); dq_minus[i] -= eps
        f_plus = ddq_numeric(q_eq, dq_plus, u_eq, params)
        f_minus = ddq_numeric(q_eq, dq_minus, u_eq, params)
        A_ddq_dq[:,i] = (f_plus - f_minus)/(2*eps)
    u_plus = u_eq + eps
    u_minus = u_eq - eps
    f_plus = ddq_numeric(q_eq, dq_eq, u_plus, params)
    f_minus = ddq_numeric(q_eq, dq_eq, u_minus, params)
    B_ddq_u[:,0] = (f_plus - f_minus)/(2*eps)
    # Construct full 6x6 A and 6x1 B
    A = np.zeros((2*n, 2*n))
    A[:n, n:] = np.eye(n) # dx/dt = dq/dt
    A[n:, :n] = A_ddq_q
    A[n:, n:] = A_ddq_dq
    B = np.zeros((2*n,1))
    B[n:,0] = B_ddq_u[:,0]
    return A, B

def controllability_matrix(A, B):
    n = A.shape[0]
    C = B
    for i in range(1,n):
        C = np.hstack((C, np.linalg.matrix_power(A,i).dot(B)))
    return C

def is_controllable(A, B):
    C = controllability_matrix(A,B)
    rank = np.linalg.matrix_rank(C)
    return rank == A.shape[0], rank
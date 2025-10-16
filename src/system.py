# src/system.py

import numpy as np

# Numeric parameters for the system
params_default = {
    'M_cart': 1.0,     # Mass of cart
    'm1': 0.3,         # Mass of first pendulum
    'm2': 0.2,         # Mass of second pendulum
    'l1': 0.5,         # Length of first pendulum
    'l2': 0.4,         # Length of second pendulum
    'lc1': 0.25,       # COM of first pendulum
    'lc2': 0.2,        # COM of second pendulum
    'I1': 0.01,        # Inertia of first pendulum
    'I2': 0.008,       # Inertia of second pendulum
    'g': 9.81          # Gravity
}

def ddq_numeric(q, dq, u, params=params_default):
    """
    Compute accelerations (ddq) given state (q,dq) and input u.
    q: [x, theta1, theta2]
    dq: [dx, dtheta1, dtheta2]
    u: scalar control input (force on cart)
    """
    # Unpack parameters
    M = params['M_cart']
    m1 = params['m1']
    m2 = params['m2']
    l1 = params['l1']
    l2 = params['l2']
    lc1 = params['lc1']
    lc2 = params['lc2']
    I1 = params['I1']
    I2 = params['I2']
    g = params['g']

    x, th1, th2 = q
    dx, dth1, dth2 = dq

    # Define inertia matrix M(q)
    M_mat = np.zeros((3,3))
    M_mat[0,0] = M + m1 + m2
    M_mat[0,1] = m1*lc1*np.cos(th1)
    M_mat[0,2] = m2*lc2*np.cos(th2)
    M_mat[1,0] = M_mat[0,1]
    M_mat[1,1] = I1 + m1*lc1**2
    M_mat[1,2] = 0
    M_mat[2,0] = M_mat[0,2]
    M_mat[2,1] = 0
    M_mat[2,2] = I2 + m2*lc2**2

    # Define generalized forces H(q,dq,u)
    H_vec = np.zeros((3,1))
    H_vec[0] = - m1*lc1*(dth1**2)*np.sin(th1) - m2*lc2*(dth2**2)*np.sin(th2) + u
    H_vec[1] = - m1*g*lc1*np.sin(th1)
    H_vec[2] = - m2*g*lc2*np.sin(th2)

    # Solve M*ddq + H = 0 => ddq = -M^-1 * H
    ddq = np.linalg.solve(M_mat, -H_vec)
    return ddq.flatten()

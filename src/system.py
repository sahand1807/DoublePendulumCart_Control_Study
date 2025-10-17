# src/system.py
import numpy as np

# Numeric parameters for the system
params_default = {
    'M_cart': 1.0,   # Mass of cart
    'm1': 0.3,       # Mass of first pendulum
    'm2': 0.2,       # Mass of second pendulum
    'l1': 0.5,       # Length of first pendulum
    'l2': 0.4,       # Length of second pendulum
    'lc1': 0.25,     # COM of first pendulum
    'lc2': 0.2,      # COM of second pendulum
    'I1': 0.01,      # Inertia of first pendulum
    'I2': 0.008,     # Inertia of second pendulum
    'g': 9.81        # Gravity
}

def ddq_numeric(q, dq, u, params=params_default):
    """
    Compute accelerations (ddq) given state (q,dq) and input u.
    
    IMPORTANT: This uses ABSOLUTE angles from vertical (as in theory.md)
    q: [x, theta1, theta2] where theta1, theta2 are ABSOLUTE from vertical
    dq: [dx, dtheta1, dtheta2]
    u: scalar control input (force on cart)
    
    Based on Lagrangian derivation in theory.md Section 2.
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

    th1 = -th1
    th2 = -th2
    dth1 = -dth1
    dth2 = -dth2
    

    # Mass/Inertia matrix M(q) from theory.md
    # Collected from kinetic energy terms
    M_mat = np.zeros((3, 3))
    
    # M[0, 0]: coefficient of dx^2
    M_mat[0, 0] = M + m1 + m2
    
    # M[0, 1]: coefficient of 2*dx*dth1
    M_mat[0, 1] = m1*lc1*np.cos(th1) + m2*l1*np.cos(th1)
    
    # M[0, 2]: coefficient of 2*dx*dth2  
    M_mat[0, 2] = m2*lc2*np.cos(th2)
    
    # M[1, 1]: coefficient of dth1^2
    M_mat[1, 1] = I1 + m1*lc1**2 + m2*l1**2
    
    # M[1, 2]: coefficient of 2*dth1*dth2 (COUPLING TERM!)
    M_mat[1, 2] = m2*l1*lc2*np.cos(th1 - th2)
    
    # M[2, 2]: coefficient of dth2^2
    M_mat[2, 2] = I2 + m2*lc2**2
    
    # Symmetry
    M_mat[1, 0] = M_mat[0, 1]
    M_mat[2, 0] = M_mat[0, 2]
    M_mat[2, 1] = M_mat[1, 2]
    
    # Coriolis/centrifugal and gravity terms
    # From Euler-Lagrange: d/dt(dL/dq_dot) - dL/dq = Q
    
    # Centrifugal terms (from velocity products)
    C_vec = np.zeros((3, 1))
    
    # Cart equation centrifugal terms
    C_vec[0] = (m1*lc1*np.sin(th1)*dth1**2 + 
                m2*l1*np.sin(th1)*dth1**2 + 
                m2*lc2*np.sin(th2)*dth2**2)
    
    # First pendulum equation
    C_vec[1] = -m2*l1*lc2*np.sin(th1 - th2)*dth2**2
    
    # Second pendulum equation  
    C_vec[2] = m2*l1*lc2*np.sin(th1 - th2)*dth1**2
    
    # Gravity terms (from potential energy gradient)
    G_vec = np.zeros((3, 1))
    G_vec[0] = 0  # No gravity on cart motion
    G_vec[1] = (m1*g*lc1 + m2*g*l1)*np.sin(th1)  # First pendulum
    G_vec[2] = m2*g*lc2*np.sin(th2)               # Second pendulum
    
    # Control input
    B_vec = np.zeros((3, 1))
    B_vec[0] = u
    B_vec[1] = 0
    B_vec[2] = 0
    
    # Equation of motion: M*ddq = B - C - G
    rhs = B_vec - C_vec - G_vec
    
    # Solve for accelerations
    ddq = np.linalg.solve(M_mat, rhs)

    ddq[1] = -ddq[1] 
    ddq[2] = -ddq[2]
    
    return ddq.flatten()
"""Newmark's Method for Single-Degree-of-Freedom (SDOF) systems."""

import numpy as np


def newmark_sdof(m: float, k: float, c: float, p: np.ndarray, dt: float, 
                 u0: float, v0: float, method: str = 'constant') -> np.ndarray:
    """
    Newmark's Method for a single-degree-of-freedom system (SDOF).
    
    Governing equation: m*u'' + c*u' + k*u = p(t)
    
    Parameters:
    -----------
    m : float
        Scalar mass.
    k : float
        Scalar stiffness.
    c : float
        Scalar damping coefficient.
    p : np.ndarray
        Vector of forcing function (must be 1D, same length as time).
    dt : float
        Scalar time step.
    u0 : float
        Initial displacement.
    v0 : float
        Initial velocity.
    method : str, optional
        'constant' for constant-average acceleration (β=1/4, γ=1/2)
        'linear' for linear acceleration (β=1/6, γ=1/2)
        Default is 'constant'.
    
    Returns:
    --------
    np.ndarray
        A 2D NumPy array where the first column is displacement (u),
        the second is velocity (v), and the third is acceleration (a).
        Shape is (n, 3), where n is the number of time steps.
    """
    # 1. Input setup
    p = np.asarray(p).flatten()  # Ensure p is a 1D NumPy array
    n = len(p)
    
    # 2. Choose Newmark parameters (beta and gamma)
    method_lower = method.lower()
    if method_lower == 'constant':
        beta = 1/4  # Constant-average acceleration
        gamma = 1/2
    elif method_lower == 'linear':
        beta = 1/6  # Linear acceleration
        gamma = 1/2
    else:
        raise ValueError(
            "newmark_sdof: Unknown method. method must be 'constant' or 'linear'."
        )
    
    # 3. Preallocate response vectors
    u = np.zeros(n)
    v = np.zeros(n)
    a = np.zeros(n)
    
    # 4. Initial conditions (at t=0, index 0 in Python)
    u[0] = u0
    v[0] = v0
    
    # Initial acceleration a(0) from EOM: m*a + c*v + k*u = p
    a[0] = (p[0] - c * v[0] - k * u[0]) / m
    
    # 5. Standard Newmark constants
    a0 = 1.0 / (beta * dt**2)
    a1 = gamma / (beta * dt)
    a2 = 1.0 / (beta * dt)
    a3 = 1.0 / (2.0 * beta) - 1.0
    a4 = gamma / beta - 1.0
    a5 = dt * (gamma / (2.0 * beta) - 1.0)
    
    # 6. Effective stiffness
    k_eff = k + a0 * m + a1 * c
    
    # 7. Time stepping loop (from time index i to i+1)
    for i in range(n - 1):
        # Effective load at step i+1
        p_eff = p[i+1] + \
                m * (a0 * u[i] + a2 * v[i] + a3 * a[i]) + \
                c * (a1 * u[i] + a4 * v[i] + a5 * a[i])
        
        # Solve for displacement at i+1
        u[i+1] = p_eff / k_eff
        
        # Update acceleration at i+1
        a[i+1] = a0 * (u[i+1] - u[i]) - a2 * v[i] - a3 * a[i]
        
        # Update velocity at i+1
        v[i+1] = v[i] + dt * ((1.0 - gamma) * a[i] + gamma * a[i+1])
    
    # Return results as a single 2D NumPy array
    return np.column_stack((u, v, a))

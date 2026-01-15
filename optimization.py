import numpy as np
import scipy.optimize
from aerodynamics import run_simulation

def objective_function(y, x):
    """The objective function to be minimized."""
    cl, cd, _, _ = run_simulation(x, y)
    if cd <= 0:
        return 1e6
    return -(cl / cd)

def optimize_airfoil(y, x, iterations=40):
    """Optimizes the airfoil shape to maximize the lift-to-drag ratio."""
    bounds = scipy.optimize.Bounds(lb=-(np.ones(y.shape)*2), ub=(np.ones(y.shape)*2), keep_feasible=False)

    result = scipy.optimize.minimize(
        fun=lambda y: objective_function(y, x),
        x0=y,
        method='Nelder-Mead',
        options={'maxiter': int(iterations)},
        bounds=bounds
    )
    return result

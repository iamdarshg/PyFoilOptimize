import os
import numpy as np
from time import time

from aerodynamics import run_simulation
from optimization import optimize_airfoil
from plotting import generate_plots
from sgd_optimizer import sgd_optimizer
from optimization import objective_function

if __name__ == '__main__':
    start_time = time()

    naca_filepath = os.path.join('0012.txt')
    with open(naca_filepath, 'r') as infile:
        x, y = np.loadtxt(infile, dtype=float, unpack=True)

    # y_optimized = sgd_optimizer(objective_function, x, y, n_iterations=10, learning_rate=0.01)

    cl, cd, panels, freestream = run_simulation(x, y)

    print(f"Lift coefficient: {cl:.4f}")
    print(f"Drag coefficient: {cd:.4f}")
    print(f"Lift-to-drag ratio: {cl/cd:.4f}")

    generate_plots(x, y, panels, freestream, cl, cd)

    end_time = time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")

import numpy as np

def sgd_optimizer(objective_function, x, y, n_iterations=100, learning_rate=0.01):
    """
    A simple implementation of Stochastic Gradient Descent.
    """
    for i in range(n_iterations):
        gradient = np.zeros_like(y)
        for j in range(len(y)):
            y_plus = y.copy()
            y_plus[j] += 1e-6
            y_minus = y.copy()
            y_minus[j] -= 1e-6

            gradient[j] = (objective_function(y_plus, x) - objective_function(y_minus, x)) / (2e-6)

        y -= learning_rate * gradient

    return y

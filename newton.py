import numdifftools as nd
import numpy as np

def optimize(x0, fun):
    """
    Perform Newton's method to find a local minimum of the function `fun`.
    :param x0: Initial guess for the minimum.
    :param fun: Function to minimize, should take a single argument.
    :return: Approximate location of the minimum.
    """
    max_iter = 1000
    x = x0

    gradient = nd.Gradient(fun)
    hessian = nd.Hessian(fun)

    for t in range(max_iter):
        gradient_value = gradient(x)
        hessian_value = hessian(x)

        if (np.isclose(np.linalg.det(hessian_value), 0)):
            raise ValueError("Second derivative is zero, cannot proceed with optimization.")

        x_next = x - np.linalg.solve(hessian_value, gradient_value)
        if np.linalg.norm(x - x_next) < 1e-4:
            return x_next
        else:
            x = x_next
    
    # if max_iter is reached without convergence
    print("Warning: Maximum iterations reached without convergence.")
    return x

optimize(2.95, np.cos)
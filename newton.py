
def approximate_derivative(fun, x, h=1e-5):
    """
    Approximate the derivative of a function at a point using central difference.
    :param fun: Function to differentiate, should take a single argument.
    :param x: Point at which to approximate the derivative.
    :param h: Step size for the approximation.
    :return: Approximate derivative of `fun` at `x`.
    """
    return (fun(x + h) - fun(x - h)) / (2 * h)

def approximate_second_derivative(fun, x, h=1e-5):
    """
    Approximate the second derivative of a function at a point using central difference.
    :param fun: Function to differentiate, should take a single argument.
    :param x: Point at which to approximate the derivative.
    :param h: Step size for the approximation.
    :return: Approximate derivative of `fun` at `x`.
    """
    return (fun(x + h) - 2 * fun(x) + fun(x - h)) / (h ** 2)

def optimize(x0, fun):
    """
    Perform Newton's method to find a local minimum of the function `fun`.
    :param x0: Initial guess for the minimum.
    :param fun: Function to minimize, should take a single argument.
    :return: Approximate location of the minimum."""
    max_iter = 1000
    x = x0

    for t in range(max_iter):
        if (approximate_second_derivative(fun, x) == 0):
            raise ValueError("Second derivative is zero, cannot proceed with optimization.")

        x = x - approximate_derivative(fun, x) / approximate_second_derivative(fun, x)
        if abs(approximate_derivative(fun, x)) < 1e-5:
            return x
        
    return x
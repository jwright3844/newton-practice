
def approximate_derivative(fun, x, h=1e-5):
    return (fun(x + h) - fun(x - h)) / (2 * h)

def approximate_second_derivative(fun, x, h=1e-5):
    return (fun(x + h) - 2 * fun(x) + fun(x - h)) / (h ** 2)

def optimize(x0, fun):
    max_iter = 1000
    x = x0

    for t in range(max_iter):
        x = x - approximate_derivative(fun, x) / approximate_second_derivative(fun, x)
        if abs(approximate_derivative(fun, x)) < 1e-5:
            return x
        
    return x
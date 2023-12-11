import numpy as np
import time

## for x in R^d
## for polynomial degree k
## For perceptron a*phi(w^T x)
## for weights w in R^d


# Polynomial of phi(x) = phi0 + phi1 x + phi2 x^2... phik x^k

class PerceptronPolynomial:
    def __init__(self, polynomial_degree, dimensions, nn_weights):
        self.polynomial_degree = polynomial_degree
        self.dimensions = dimensions
        self.nn_weights = nn_weights

    def get_polynomial_weights(self, activation_polynomial_weights):
        ## Activation polynomial weights as [1, x, x^2, ...]
        shape = (self.polynomial_degree,)
        for i in range(self.dimensions):
            shape = shape + (self.polynomial_degree,)

        def fill_weights_tensor(degrees, weights_tensor):
            total_degree = int(np.sum(degrees))


        weights_tensor = np.empty(shape)
        do_for_polynomial_combinations(dimensions=self.dimensions,
                                       order=self.polynomial_degree,
                                       action=1,
                                       target=None)

def do_for_polynomial_combinations(dimensions, order, action, target):
    # 10, 10 takes 1 second
    _do_for_polynomial_combinations_recursive(np.array([]), dimensions, order, action, target)

def _do_for_polynomial_combinations_recursive(fixed_dimensions, total_dimensions, order, action, target):
    if len(fixed_dimensions) == total_dimensions:
        action(fixed_dimensions, target)
    else:
        current_order = int(np.sum(fixed_dimensions))
        for new_order in range(order - current_order+1):
            new_fixed_dimensions = np.copy(fixed_dimensions).astype(np.int16)
            new_fixed_dimensions = np.append(new_fixed_dimensions, new_order)
            _do_for_polynomial_combinations_recursive(new_fixed_dimensions, total_dimensions, order, action, target)

def print_dimensions(x, target):
    s = ""
    for i in range(len(x)):
        s += "," + str(x[i])
    # print(s)

def get_total_num(x, target):
    target += 1


if __name__=="__main__":
    start = time.perf_counter()
    target = np.array([0])
    do_for_polynomial_combinations(dimensions=10, order=5, action=get_total_num, target=target)
    print(target)
    print(time.perf_counter() - start)

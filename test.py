from simulator import spc_ansatz, pauli_ansatz, get_gradient_fd, get_gradient_ps
from math import comb
import numpy as np
import unittest
from parameterized import parameterized


np.random.seed(42)

test_cases = [[[n, k] for k in range(1, n)] for n in [2, 4, 6]]
test_cases = [item for sublist in test_cases for item in sublist]

class SPCTest(unittest.TestCase):

    @parameterized.expand(test_cases)
    def test_construction(self, n, m):
        num_pars = 2*comb(n, m) - 2
        pars = np.random.uniform(-np.pi, +np.pi, size=num_pars)
        ans = spc_ansatz(n, m)
        ans(pars)
    
    @parameterized.expand(test_cases)
    def test_gradient_methods(self, n, k):
        axes = np.random.choice([0, 1, 2, 3], size=(k, n))
        pars = np.random.uniform(-np.pi, +np.pi, size=k)
        op = np.random.uniform(-1, 1, size=(2**n, 2**n))

        ans = pauli_ansatz(axes, None)

        grad_fd = get_gradient_fd(ans, op, pars, epsilon=1e-12)
        grad_ps = get_gradient_ps(ans, op, pars)

        np.testing.assert_allclose(grad_fd,grad_ps, rtol=1e-2)


if __name__ == '__main__':
    unittest.main()
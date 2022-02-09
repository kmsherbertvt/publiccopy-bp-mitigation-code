from simulator import spc_ansatz, pauli_ansatz, get_gradient_fd, get_gradient_ps, aswap_gate
from math import comb
import numpy as np
import unittest
import traceback
from parameterized import parameterized

np.random.seed(42)
test_cases = [[[n, k] for k in range(1, n)] for n in [2, 4, 6]]
test_cases = [item for sublist in test_cases for item in sublist]


def statistical_test(num_tests: int, fraction_passes: float):
    """Wrapper for statistical test that should be run multiple times and pass
    at least a certain fraction of times.
    """
    def stat_test(func):
        def wrapper_func(*args, **kwargs):
            num_failures = 0
            num_passes = 0
            for _ in range(num_tests):
                try:
                    func(*args, **kwargs)
                    num_passes += 1
                except Exception as e:
                    print(traceback.format_exc())
                    print(f'Exception found: {e}')
                    num_failures += 1
            if num_passes / num_tests < fraction_passes:
                raise ValueError(f'Passed {num_passes} out of {num_tests} trials, needed {100 * fraction_passes}')
        return wrapper_func
    return stat_test


def hamming_weight(n: int) -> int:
    c = 0
    while n:
        c += 1
        n &= n - 1
    return c


def bitmask(n: int, k: int, inverse_mask: bool = False) -> np.array:
    res = []
    for i in range(2**n):
        if hamming_weight(i) == k:
            if inverse_mask:
                res.append(0)
            else:
                res.append(1)
        else:
            if inverse_mask:
                res.append(1)
            else:
                res.append(0)
    vec_out = np.array(res).reshape(2**n, 1)
    return vec_out


class SimulatorTest(unittest.TestCase):

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
    
    @statistical_test(50, 0.99)
    def test_particle_number(self):
        for n, k in test_cases:
            ans = spc_ansatz(n, k)
            num_pars = 2*comb(n, k) - 2
            pars = np.random.uniform(-np.pi, +np.pi, size=num_pars)
            vec = ans(pars)

            masked = vec * bitmask(n, k, inverse_mask=True)
            norm = np.linalg.norm(masked)

            if norm > 1e-2:
                raise ValueError(f'Particle number not preserved, norm of masked vec={norm}')


if __name__ == '__main__':
    unittest.main()
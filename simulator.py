import numpy as np
from functools import lru_cache, reduce
from math import comb
from scipy.linalg import kron
from numpy.linalg import multi_dot
import numba
from numpy import einsum
from opt_einsum import contract
from typing import List, Union, Optional, Tuple, Callable
from kron_vec_product import kron_vec_prod

Array = np.array

Ansatz = Callable[
    [Union[Array, List[float]]],
    Array
]

#@lru_cache(4)
def get_pauli(i: int) -> Array:
    if i == 0:
        out = np.eye(2) * (1.0+0.j)
    elif i == 1:
        out = np.array([[0.0, 1.0], [1.0, 0.0]]) * (1.0+0.j)
    elif i == 2:
        out = np.array([[0.0, -1j], [1j, 0.0]]) * (1.0+0.j)
    elif i == 3:
        out = np.array([[1.0, 0.0], [0.0, -1.0]]) * (1.0+0.j)
    return out

#@numba.jit(nopython=True, fastmath=True)
def rot_p(theta: float, a: int) -> np.array:
    if a == 1 or a == 2:
        c = np.cos(theta/2)
        s = np.sin(theta/2)
        if a == 1:
            return np.array([[c, -1j*s], [-1j*s, c]]) * (1.0 + 0.j)
        else:
            return np.array([[c, -s], [s, c]]) * (1.0 + 0.j)
    elif a == 3:
        p = np.exp(-1j*theta/2)
        return np.array([[p, 0.0 + 0.j], [0.0 + 0.j, 1/p]])

def kron_args(*args):
    """For `args = [m1, m2, m3, ...]`, return
    `m1 otimes m2 otimes m3 otimes ...`.
    """
    return reduce(np.kron, args)

@lru_cache(200)
def pauli_str(axes: Tuple[int]) -> Array:
    res = get_pauli(axes[0])
    for ax in axes[1:]:
        res = kron(res, get_pauli(ax))
    return res

#@lru_cache(5)
def id_mat(n: int) -> Array:
    return np.eye(2**n)

@numba.jit(nopython=True, fastmath=True)
def pauli_exp_rule(identity: Array, mat: Array, theta: float) -> Array:
    return np.cos(theta) * identity - 1j * np.sin(theta) * mat

def pauli_exp(mat: Array, theta: float) -> Array:
    """Given a Hermitian, Unitary operator H and angle theta,
    return U = e^{-i theta H} = cos(theta)*Id - 1j*sin(theta)*H
    """
    n = int(np.log2(mat.shape[0]))
    identity = id_mat(n)
    return pauli_exp_rule(identity, mat, theta)

def mcp_g_list(n: int) -> List[List[int]]:
    res = []
    for i in range(n):
        ax = [0] * n
        ax[i] = 2
        res.append(ax)
    
    for i in range(n-1):
        ax = [0] * n
        ax[i] = 3
        ax[i+1] = 2
        res.append(ax)
    
    return res

def pauli_ansatz(axes: List[int], initial_state: Array = None) -> Ansatz:
    num_pars = len(axes)
    num_qubits = len(axes[0])
    identity = id_mat(num_qubits)
    if initial_state is None:
        initial_state = zero_state(num_qubits)
    def _ans_out(pars: List[float]) -> Array:
        if len(pars) != num_pars:
            raise ValueError('Invalid number of parameters supplied')
        current_state = initial_state
        for theta, op in zip(pars, axes):
            mat = pauli_str(tuple(op))
            p = pauli_exp_rule(identity, mat, theta)
            current_state = einsum('ab,bc->ac', p, current_state)
        return current_state
    return _ans_out

def aswap_gate(theta: float, phi: float) -> Array:
    a = np.zeros((4, 4), dtype=complex)
    c = np.cos(theta)
    s = np.sin(theta)
    p = np.exp(1j*phi)
    a[0, 0] = 1.0 + 0.j
    a[3, 3] = 1.0 + 0.j
    a[1, 1] = c
    a[2, 2] = -c
    a[1, 2] = s*p
    a[2, 1] = s/p
    return a

def zero_state(n: int) -> Array:
    res = np.zeros((2**n, 1)) * (1.0 + 0.j)
    res[0, 0] = 1.0
    return res

def hwe_ansatz(num_qubits: int, depth: int) -> Ansatz:
    if num_qubits % 2 != 0:
        raise ValueError('Only even number of qubits supported')
    initial_state = zero_state(num_qubits)

    # Make V
    cz = np.diag([1, 1, 1, -1])
    V = kron_args(*[cz] * (num_qubits // 2))
    v_list = [np.eye(2)] + [cz] * ((num_qubits // 2) - 1) + [np.eye(2)]
    V = kron_args(*v_list).dot(V)
    axes = np.random.choice(list(range(1, 4)), size=(num_qubits, depth))

    def _ans_out(pars: List[float]) -> Array:
        pars = list(pars)
        state = kron_args(*[rot_p(np.pi/4, 2)] * num_qubits).dot(initial_state)
        if len(pars) != num_qubits * depth:
            raise ValueError('Invalid number of pars')
        for d in range(depth):
            rots = []
            for i in range(num_qubits):
                a = axes[i, d]
                rots.append(rot_p(pars.pop(), a))
            U = kron_args(*rots)
            state = einsum('ab,bc->ac', U, state)
            state = einsum('ab,bc->ac', V, state)
        return state
    
    return _ans_out

def spc_ansatz(num_qubits: int, num_particles: int) -> Ansatz:
    if num_qubits % 2 == 1:
        raise ValueError('Number of qubits must be even')
    if num_particles > num_qubits or num_particles < 0:
        raise ValueError('Invalid number of particles given')
    num_pars = 2*(comb(num_qubits, num_particles) - 1)
    init_occs = [get_pauli(0)] * (num_qubits-num_particles) + [get_pauli(1)] * num_particles
    init_gates = kron_args(*init_occs)
    initial_state = init_gates.dot(zero_state(num_qubits))

    def _schedule_even(
        theta_pars: List[float], 
        phi_pars: List[float],
        num_qubits: int
    ) -> List[Array]:
        schedule = []

        shifted_layer = True
        while len(theta_pars) > 0:
            shifted_layer = not shifted_layer
            layer = []
            if shifted_layer:
                layer.append(id_mat(1))
            
            for _ in range(num_qubits//2 - 1 if shifted_layer else num_qubits//2):
                if len(theta_pars) == 0:
                    break
                t, p = theta_pars.pop(), phi_pars.pop()
                layer.append(aswap_gate(t, p))
            
            if len(theta_pars) == 0:
                break
            if shifted_layer:
                layer.append(id_mat(1))

            schedule.append(layer)
        
        if shifted_layer and len(layer) == num_qubits + 1:
            schedule.append(layer)
            return schedule
        elif (not shifted_layer) and len(layer) == num_qubits //2:
            schedule.append(layer)
            return schedule
        else:
            pass
        
        current_num_qubits = 0
        for mat in layer:
            current_num_qubits += int(np.log2(mat.shape[0]))

        remaining_qubits = num_qubits - current_num_qubits
        
        layer.extend([id_mat(1)]*remaining_qubits)
        schedule.append(layer)
        
        return schedule

    def _ans_out(pars: List[float]) -> Array:
        """
        pars = [theta1, ..., thetak, phi1, ..., phik]
        """
        pars = list(reversed(pars))
        if len(pars) != num_pars:
            raise ValueError('Invalid number of pars given')
        k = len(pars) // 2
        theta_pars = pars[0:k]
        phi_pars = pars[k:2*k]

        current_state = initial_state

        schedule = _schedule_even(theta_pars, phi_pars, num_qubits)

        for layer in schedule:
            mat = kron_args(*layer)
            current_state = mat.dot(current_state)

        return current_state
    return _ans_out
        

@numba.jit(nopython=True, fastmath=True)
def exp_val(op: Array, state: Array) -> float:
    res = state.conj().transpose().dot(op).dot(state)[0, 0]
    return res


def get_gradient_ps(ans: Ansatz, op: np.array, point: np.array, r: float = 1) -> np.array:
    point = np.array(point)
    k = len(point)
    grad = []

    for i in range(k):
        mask = np.zeros(k)
        mask[i] = np.pi/4

        state_plus = ans(point + mask)
        state_minus = ans(point - mask)

        df = exp_val(op, state_plus) - exp_val(op, state_minus)
        df *= r

        grad.append(df)

    return np.array(grad)

def get_gradient_fd(ans: Ansatz, op: np.array, point: np.array, epsilon: float = 1e-12, grad_pars: List[int] = None) -> np.array:
    point = np.array(point)
    k = len(point)

    if grad_pars is None:
        grad_pars = range(k)
    else:
        if not set(grad_pars).issubset(range(k)):
            raise ValueError(f'Given list of parameters {grad_pars} is not a subset of {range(k)}')

    grad = []
    for i in grad_pars:
        mask = np.zeros(k)
        mask[i] = epsilon/2

        state_plus = ans(point + mask)
        state_minus = ans(point - mask)

        df = exp_val(op, state_plus) - exp_val(op, state_minus)
        df /= epsilon

        grad.append(df)

    return np.array(grad)
import numpy as np
from math import comb
from scipy.linalg import kron
from typing import List, Union, Optional, Tuple, Callable

Array = np.array

Ansatz = Callable[
    [Union[Array, List[float]]],
    Array
]

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

def kron_args(*args):
    """For `args = [m1, m2, m3, ...]`, return
    `m1 otimes m2 otimes m3 otimes ...`.
    """
    if len(args) == 1:
        return args[0]
    res = args[0]
    for m in args[1:]:
        res = kron(res, m)
    return res

def pauli_str(axes: Tuple[int]) -> Array:
    res = get_pauli(axes[0])
    for ax in axes[1:]:
        res = kron(res, get_pauli(ax))
    return res

def id_mat(n: int) -> Array:
    return np.eye(2**n)

def pauli_exp_rule(identity: Array, mat: Array, theta: float) -> Array:
    return np.cos(theta) * identity - 1j * np.sin(theta) * mat

def pauli_exp(mat: Array, theta: float) -> Array:
    """Given a Hermitian, Unitary operator H and angle theta,
    return U = e^{-i theta H} = cos(theta)*Id - 1j*sin(theta)*H
    """
    n = int(np.log2(mat.shape[0]))
    identity = id_mat(n)
    return pauli_exp_rule(identity, mat, theta)

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
            mat = pauli_str(op)
            current_state = pauli_exp_rule(identity, mat, theta).dot(current_state)
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
    a[2, 3] = s*p
    a[3, 2] = s/p
    return a

def zero_state(n: int) -> Array:
    res = np.zeros((2**n, 1)) * (1.0 + 0.j)
    res[0, 0] = 1.0
    return res

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

def get_gradient_fd(ans: Ansatz, op: np.array, point: np.array, epsilon: float = 1e-8, grad_pars: List[int] = None) -> np.array:
    point = np.array(point)
    k = len(point)
    grad = []

    grad_pars = range(k) if grad_pars is None else grad_pars

    for i in grad_pars:
        mask = np.zeros(k)
        mask[i] = epsilon/2

        state_plus = ans(point + mask)
        state_minus = ans(point - mask)

        df = exp_val(op, state_plus) - exp_val(op, state_minus)
        df /= epsilon

        grad.append(df)

    return np.array(grad)
import numpy as np
import pandas as pd
import logging
from scipy import optimize
from typing import Callable, Dict, List

from callbacks import Callback, MaxGradientStopper

from simulator import (
    get_pauli,
    pauli_str,
    pauli_exp_rule,
    pauli_exp,
    pauli_ansatz,
    zero_state,
    plus_state,
    exp_val,
    get_gradient_fd,
    make_connectivity_pool,
    make_complete_pool
)


class ADAPTResult:

    def __init__(self):
        self._result_list = []
        self._halt_reason = 'None'
    
    @property
    def iteration(self):
        return len(self._result_list)
    
    @property
    def halt_reason(self):
        return self._halt_reason

    def add_step_res(
        self,
        energy: float,
        gradients: np.array,
        parameters: np.array,
        op_indices: List[int],
        state: np.array = None
    ):
        max_grad = np.array([np.max(np.abs(gradients))])
        self._result_list.append({
            'energy': energy.real,
            'gradients': list(gradients),
            'opt_params': list(parameters),
            'min_vector': state,
            'max_grad': max_grad,
            'op_indices': list(op_indices.copy())
        })
    
    @property
    def dataframe(self):
        return pd.DataFrame(self._result_list)
    
    @property
    def result(self):
        if self.iteration == 0:
            raise ValueError('ADAPT has not been run yet')
        return self._result_list[-1]
    
    @property
    def step_history(self):
        return self._result_list


def _mat_mul(a, b):
    return a.dot(b)

def adapt(
    ham: np.array,
    pool: List[np.array],
    optimizer: Callable,
    callbacks: List[Callback],
    optimizer_args: Dict = None,
    initial_state: np.array = None,
) -> ADAPTResult:
    
    if optimizer_args is None:
        opt_args = {}
    else:
        opt_args = optimizer_args.copy()

    if initial_state is None:
        initial_state = zero_state(len(pool[0])) * (1.0 + 0.j)

    # Compute commutators
    logging.info('Computing commutators')
    pool_mats = [pauli_str(tuple(axes)) for axes in pool]
    comms = [_mat_mul(ham, op) - _mat_mul(op, ham) for op in pool_mats]
    del pool_mats

    # Initialize result object
    result = ADAPTResult()

    # Convention: e^{-i theta H} for operators
    init_grads = [-1j * exp_val(com, initial_state) for com in comms]
    init_grads = np.real(init_grads)

    # Add step before potentially immediately exiting
    result.add_step_res(
        energy=exp_val(ham, initial_state),
        gradients=init_grads,
        parameters=np.array([]),
        op_indices=[]
    )

    # Check if immediately converged
    for callback in callbacks:
        if isinstance(callback, MaxGradientStopper):
            if callback.halt(result.step_history):
                result._halt_reason = callback.halt_reason(
                    result.step_history
                )
                return result
    
    # Initialize outer loop variables
    ansatz_op_inds = [np.argmax(init_grads)]
    ansatz_ops = [pool[i] for i in ansatz_op_inds]
    opt_pars = np.array([0.0])
    ans = pauli_ansatz(ansatz_ops, initial_state=initial_state)

    def _fn(pars) -> float:
        return exp_val(ham, ans(pars)).real
    
    if 'constrained' in opt_args.keys():
        opt_args.pop('constrained')
        constrained = True
        raise ValueError('Not supported yet')
    else:
        constrained = False
    
    optimization_result = optimizer(_fn, x0=opt_pars, **opt_args) # type: optimize.OptimizeResult
    opt_pars = optimization_result.x
    state = ans(opt_pars)
    gradients = [-1j * exp_val(com, state) for com in comms]
    gradients = np.array(gradients).real

    result.add_step_res(
        energy=optimization_result.fun,
        gradients=gradients,
        parameters=opt_pars,
        op_indices=ansatz_op_inds,
        state=None
    )

    while True:
        for callback in callbacks:
            if callback.halt(result.step_history):
                result._halt_reason = callback.halt_reason(
                    result.step_history
                )
                return result
        
        ansatz_op_inds.append(np.argmax(gradients))
        ansatz_ops.append(pool[np.argmax(gradients)])
        opt_pars = np.concatenate([opt_pars, np.array([0.0])])
        ans = pauli_ansatz(ansatz_ops, initial_state=initial_state)

        def _fn(pars) -> float:
            return exp_val(ham, ans(pars)).real
        
        optimization_result = optimizer(_fn, x0=opt_pars, **opt_args) # type: optimize.OptimizeResult
        opt_pars = optimization_result.x
        state = ans(opt_pars)
        gradients = [-1j * exp_val(com, state) for com in comms]
        gradients = np.array(gradients).real

        result.add_step_res(
            energy=optimization_result.fun,
            gradients=gradients,
            parameters=opt_pars,
            op_indices=ansatz_op_inds,
            state=None
        )
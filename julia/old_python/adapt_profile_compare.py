import numpy as np
from functools import lru_cache
from scipy.optimize import minimize
from math import comb
from dask.distributed import Client, progress, wait
from itertools import product
from typing import List, Dict, Tuple
from math import comb
from dask_jobqueue import SLURMCluster
from dask import delayed
from distributed import Future
from distributed import as_completed
from simulator import filename_format
from dask.dataframe import from_pandas, from_delayed
from random import shuffle
import dask
import networkx as nx
import dask.bag as db
import pandas as pd
import time
import logging

import time

from adapt import ADAPTResult, adapt

from callbacks import (
    ParameterStopper,
    FloorStopper,
    NoImprovementStopper,
    DeltaYStopper
)

from simulator import (
    Array,
    pauli_str,
    get_gradient_fd,
    spc_ansatz,
    pauli_ansatz,
    mcp_g_list,
    make_complete_pool,
    hwe_ansatz,
    two_local_pool,
    qaoa_ansatz,
    make_connectivity_pool,
    plus_state
)

logging.basicConfig(
    filename='adapt_benchmark.log',
    level=logging.INFO
)

def random_graph(n: int, p: float) -> nx.Graph:
    g = nx.generators.random_graphs.random_regular_graph(3, n)
    for i, j in g.edges:
        g[i][j]['weight'] = 1.0
    return g


def get_max_cut_ham(g: nx.Graph) -> np.array:
    n = len(g)
    ham = np.zeros((2**n, 2**n), dtype='complex128')
    iden = np.eye(2**n, dtype='complex128')
    for i, j in g.edges:
        l = [0] * n
        l[i] = 3
        l[j] = 3
        part = iden - pauli_str(axes=tuple(l))
        part *= g[i][j]['weight']
        part *= (1.0 + 0.j)
        ham += part
    ham /= -2.0
    return ham


def adapt_run(d: Dict) -> Dict:
    np.random.seed(d['seed'])
    if d['pool_name'] == 'path':
        graph = nx.path_graph(d['n'])
        pool = make_connectivity_pool(graph)
    elif d['pool_name'] == 'two-local':
        pool = two_local_pool(d['n'])
    else:
        raise NotImplementedError
    
    def optimizer(*args, **kwargs):
        return minimize(*args, method='BFGS', **kwargs)
    
    ham = get_max_cut_ham(random_graph(d['n'], p=3))
    e_min = np.min(np.diagonal(ham))
    e_max = np.max(np.diagonal(ham))
    gap = e_max - e_min

    callbacks = [
        ParameterStopper(20),
    ]
    result = adapt(
        ham=ham,
        pool=pool,
        optimizer=optimizer,
        callbacks=callbacks,
        initial_state=plus_state(d['n']),
        new_parameter=1e-5
    )
    energy = result.step_history[-1]['energy']
    energy_error = energy - e_min

    d_out = {
        'n': d['n'],
        'pool_name': d['pool_name'],
        'e_min': e_min,
        'e_max': e_max,
        'gap': gap,
        'energy': energy,
        'energy_error': energy_error,
        'relative_err': energy_error / gap,
        'max_grads': [res['max_grad'][0] for res in result.step_history],
        'halt_reason': result.halt_reason
    }
    del result
    del ham
    try:
        del graph
    except:
        pass
    del pool
    
    return d_out

if __name__ == '__main__':
    for n in [4, 6, 8]:
        print(f"Num qubits: {n}")
        for s in range(4):
            t_0 = time.time()
            res = adapt_run({
                'seed': s,
                'pool_name': 'two-local',
                'n': n
            })
            en_err = res['relative_err']
            print(f"Energy error: {en_err}")
            delta_t = time.time() - t_0
            print(f"Elapsed time: {delta_t}")
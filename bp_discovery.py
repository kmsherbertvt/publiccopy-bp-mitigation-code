import numpy as np
from functools import lru_cache
from math import comb
from dask.distributed import Client, progress, wait
from typing import List, Dict
from math import comb
from dask_jobqueue import SLURMCluster
from dask import delayed
from distributed import Future
from distributed import as_completed
from simulator import filename_format
from dask.dataframe import from_pandas, from_delayed
from random import shuffle
import dask
import dask.bag as db
import pandas as pd
import time
import logging

import time

from simulator import (
    Array,
    pauli_str,
    get_gradient_fd,
    spc_ansatz,
    pauli_ansatz,
    mcp_g_list,
    hwe_ansatz,
    qaoa_ansatz
)

logging.basicConfig(
    filename='bp_discovery.log',
    level=logging.INFO
)


def get_op(n: int) -> Array:
    """Returns Ising Hamiltonian on n qubits. Constants = 1.
    """
    res = np.zeros((2**n, 2**n), dtype='complex128')
    for i in range(n-1):
        l = [0] * n
        l[i] = 3
        res += pauli_str(axes=tuple(l))
        l[i+1] = 3
        res += pauli_str(axes=tuple(l))
    l = [0] * n
    l[n-1] = 3
    res += pauli_str(axes=tuple(l))
    return res


def gen_rand_exps(
    qubits_range: List[int],
    num_samples: int,
    layers: List[int]
):
    inputs = []
    logging.info('Defining experiments for random circs')
    for n in qubits_range:
        for l in layers:
            axes = [
                np.random.choice([0, 1, 2, 3], size=n)
                for _ in range(l)
            ]
            for _ in range(num_samples):
                point = np.random.uniform(-np.pi, +np.pi, size=l)
                k = np.random.randint(0, l)
                inputs.append({'n': n, 'axes': axes, 'point': point, 'k': k, 'ans': 'rand'})
    logging.info(f'Defined {len(inputs)} experiments')
    return inputs


def gen_mcp_exps(
    qubits_range: List[int],
    num_samples: int,
    depth_range: List[int]
):
    inputs = []
    logging.info('Defining experiments for MCP circs')
    for n in qubits_range:
        pool = mcp_g_list(n)
        for l in depth_range:
            num_pars = l
            for _ in range(num_samples):
                axes_inds = list(np.random.choice(range(len(pool)), size=l))
                axes = [pool[i] for i in axes_inds]
                k = np.random.randint(0, l)
                inputs.append({'n': n, 'l': l, 'axes': axes, 'k': k, 'ans': 'mcp', 'num_pars': num_pars})
    logging.info(f'Defined {len(inputs)} experiments')
    return inputs


def gen_hwe_exps(
    qubits_range: List[int],
    num_samples: int,
    depth_range: List[int]
):
    inputs = []
    logging.info('Defining experiments for HWE circs')
    for n in qubits_range:
        for l in depth_range:
            num_pars = l*n
            for _ in range(num_samples):
                k = np.random.randint(0, l*n)
                inputs.append({'n': n, 'l': l, 'k': k, 'ans': 'hwe', 'num_pars': num_pars})
    logging.info(f'Defined {len(inputs)} experiments')
    return inputs


def gen_qaoa_exps(
    qubit_range: List[int],
    num_samples: int,
    depth_range: List[int],
    graph_types: List[str] = None
):
    inputs = []
    if graph_types is None:
        graph_types = 'random'
    logging.info('Defining experiments for QAOA circs')
    for n in qubit_range:
        for p in depth_range:
            for graph in graph_types:
                H = np.random.uniform(-1, 1, size=2**n)
                for _ in range(num_samples):
                    inputs.append({
                        'n': n,
                        'p': p,
                        'graph': graph,
                        'H': H,
                        'ans': 'qaoa',
                        'num_pars': 2*p,
                        'k': np.random.randint(0, 2*p)
                    })
    logging.info(f'Defined {len(inputs)} experiments')
    return inputs


def gen_spc_exps(
    qubits_range: List[int],
    num_samples: int,
    time_reversal_symmetry: bool = True
):
    if not time_reversal_symmetry:
        raise ValueError('How about no')
    inputs = []
    logging.info('Defining experiments for SPC circs')
    for n in qubits_range:
        for m in range(1, n):
            num_pars = comb(n, m) - 1
            for _ in range(num_samples):
                if time_reversal_symmetry:
                    k = np.random.randint(0, num_pars)
                else:
                    k = np.random.randint(0, 2*num_pars)
                theta_pars = np.random.uniform(-np.pi, +np.pi, size=num_pars)

                if time_reversal_symmetry:
                    phi_pars = np.zeros(len(theta_pars))
                else:
                    phi_pars = np.random.uniform(-np.pi, +np.pi, size=num_pars)
                """ Before I forgot to change this back to theta_pars, phi_pars, and I was using
                theta_pars, theta_pars. So at some point I need to re-run this, but I don't think
                it will change the result.
                """
                point = np.concatenate((theta_pars, phi_pars))
                inputs.append({'n': n, 'm': m, 'point': point, 'k': k, 'ans': 'spc', 'time_rev_sym': time_reversal_symmetry})
    logging.info(f'Defined {len(inputs)} experiments')
    return inputs


def grad_comp(d: Dict) -> float:
    ans_name = d['ans']
    op = get_op(d['n'])
    n = d['n']
    num_pars = d['num_pars']

    point = np.random.uniform(-np.pi, +np.pi, size=num_pars)

    if ans_name == 'rand':
        ans = pauli_ansatz(axes=d['axes'])
    elif ans_name == 'mcp':
        ans = pauli_ansatz(axes=d['axes'])
    elif ans_name == 'spc':
        ans = spc_ansatz(num_qubits=n, num_particles=d['m'])
    elif ans_name == 'hwe':
        ans = hwe_ansatz(num_qubits=n, depth=d['l'])
    elif ans_name == 'qaoa':
        ans = qaoa_ansatz(d['H'], p=d['p'])
    else:
        raise ValueError(f'Invalid ansatz given: {ans_name}')
    
    logging.info(f'Computing gradient on {n} qubits')
    grad = get_gradient_fd(ans=ans, op=op, point=point, grad_pars=[d['k']]).real
    
    d_out = d
    d_out['grad'] = grad[0]
    d_out.pop('point', None)
    d_out.pop('axes', None)
    d_out.pop('H', None)
    del ans
    del op
    del point
    return d_out


def grad_futures(l: List[dict], client: Client) -> List[Future]:
    logging.info('Submitting futures')
    futures = client.map(grad_comp, l, retries=1)
    logging.info('Submitted futures')
    return futures

if __name__ == '__main__':
    logging.info('Starting dask cluster/client...')
    t_o = time.time()
    np.random.seed(42)

    cluster = SLURMCluster(extra=["--lifetime-stagger", "2m"])
    logging.info('Scaling cluster...')
    cluster.scale(jobs=20)
    client = Client(cluster)
    client.upload_file('simulator.py')

    # Define experiments
    num_samples = 1500
    qubits = [4, 6, 8]
    depth_range = [1, 2, 3, 4, 5, 10, 15, 20, 25, 50, 75, 100]

    experiments = []
    #experiments.extend(gen_spc_exps(qubits, num_samples))
    #experiments.extend(gen_mcp_exps(qubits, num_samples, depth_range=depth_range))
    #experiments.extend(gen_hwe_exps(qubits, num_samples, depth_range=depth_range))
    experiments.extend(gen_qaoa_exps(qubits, num_samples, depth_range))
    shuffle(experiments)

    futures = grad_futures(experiments, client=client)
    res_list = []
    failed_futures = []
    for future in as_completed(futures, with_results=False):
        if future.status == 'error':
            logging.info(f'Future failed: {future.exception()}')
            failed_futures.append(future)
        else:
            res_list.append(future.result())
            del future
    logging.info(f'Completed running futures, {len(failed_futures)} failed')
    logging.info('Making dataframe...')
    df = pd.DataFrame(res_list)
    logging.info('Writing to disk')
    df.to_csv('data/'+filename_format(__file__, 'qaoa'))
    logging.info('Wrote to disk, exiting!')
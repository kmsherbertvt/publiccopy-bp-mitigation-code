import numpy as np
from functools import lru_cache
from math import comb
from dask.distributed import Client, progress, wait
from typing import List, Dict
from math import comb
from dask_jobqueue import SLURMCluster
from dask import delayed
from dask.dataframe import from_pandas, from_delayed
from random import shuffle
import dask
import dask.bag as db
import pandas as pd
import time
import logging

from simulator import (
    Array,
    pauli_str,
    get_gradient_fd,
    spc_ansatz,
    pauli_ansatz,
    mcp_g_list
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
        res += pauli_str(axes=l)
        l[i+1] = 3
        res += pauli_str(axes=l)
    l = [0] * n
    l[n-1] = 3
    res += pauli_str(axes=l)
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
            for _ in range(num_samples):
                point = np.random.uniform(-np.pi, +np.pi, size=l)
                axes_inds = list(np.random.choice(range(len(pool)), size=l))
                axes = [pool[i] for i in axes_inds]
                k = np.random.randint(0, l)
                inputs.append({'n': n, 'l': l, 'point': point, 'axes': axes, 'k': k, 'ans': 'mcp'})
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
                point = np.concatenate((theta_pars, theta_pars))
                inputs.append({'n': n, 'm': m, 'point': point, 'k': k, 'ans': 'spc', 'time_rev_sym': time_reversal_symmetry})
    logging.info(f'Defined {len(inputs)} experiments')
    return inputs


def grad_comp(d: Dict) -> float:
    ans_name = d['ans']
    op = get_op(d['n'])
    n = d['n']

    if ans_name == 'rand':
        ans = pauli_ansatz(axes=d['axes'])
    elif ans_name == 'mcp':
        ans = pauli_ansatz(axes=d['axes'])
    elif ans_name == 'spc':
        ans = spc_ansatz(num_qubits=n, num_particles=d['m'])
    else:
        raise ValueError(f'Invalid ansatz given: {ans_name}')
    
    logging.info(f'Computing gradient on {n} qubits')
    grad = get_gradient_fd(ans=ans, op=op, point=d['point']).real
    
    d_out = d
    d_out['grad'] = grad
    return d_out


def experiments_to_dataframe(exps: List[Dict]):
    logging.info('Shuffling list...')
    shuffle(exps)

    """ Neither of these approaches work for some reason. In each case
    they stop on the last line of the commented blocks.
    """
    #logging.info('Make pandas df')
    #df_pd = pd.DataFrame(exps)
    #logging.info('Make dask df from df_pd')
    #df = from_pandas(df_pd, npartitions=800, sort=False)
    #logging.info('Apply gradient comp')
    #df['grad'] = df.apply(lambda row: grad_comp(dict(row)), axis=1)

    logging.info('Creating bag...')
    bag = db.from_sequence(exps, npartitions=800)
    logging.info('Mapping bag with grad_comp')
    bag = bag.map(grad_comp)
    logging.info('Creating dask DataFrame from bag')
    df = bag.to_dataframe()

    logging.info('Exploding gradients')
    df = df.explode('grad')
    logging.info('Modifying gradient type')
    df['grad'] = df['grad'].astype(float)
    logging.info('Returning dask DataFrame...')
    logging.info(f'DataFrame has columns: {df.columns}')
    return df


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
    num_samples = 500
    qubits = [4, 6, 8, 10]

    experiments = []
    experiments.extend(gen_spc_exps(qubits, num_samples))
    #experiments.extend(gen_mcp_exps(qubits, num_samples, depth_range=[100, 500, 1000, 1500, 2000]))

    logging.info('Creating DataFrame with experiments')
    df = experiments_to_dataframe(experiments)
    logging.info('DataFrame defined...')

    # Do statistics
    """ Some points in the SPCs have exactly zero gradient. I think this is either due to the
    first few initial gates potentially acting trivially, but it may be due to some more fundamental
    symmetry. For now I am just discarding the points that have *exactly* zero gradient.
    """
    logging.info('Excluding zero-gradient points')
    df = df[df['grad'] != 0.0]

    logging.info('Defining pivot...')
    pivot = df.groupby(['n', 'm']).agg({'grad': ['mean', 'std']})
    logging.info('Dumping to file...')
    pivot.to_csv('data/spc/output_*.csv')
    logging.info('Dumped to file...')

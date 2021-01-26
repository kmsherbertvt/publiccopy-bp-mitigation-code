import numpy as np
from math import comb
from dask.distributed import Client, progress, wait
from typing import List
from math import comb
from dask_jobqueue import SLURMCluster
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
    """Returns Z1 Z2 on `n` qubits.
    """
    if n <= 1:
        raise ValueError('Must be at least 2 qubits')
    l = [0] * n
    l[0] = 3
    l[1] = 3
    return pauli_str(axes=l)


def gen_random_circs(
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
            for i in range(num_samples):
                point = np.random.uniform(-np.pi, +np.pi, size=l)
                inputs.append({'n': n, 'axes': axes, 'point': point})
    logging.info(f'Defined {len(inputs)} experiments')

    # Shuffle so jobs are more homogeneous
    logging.info('Shuffling experiments...')
    shuffle(inputs)

    def grad_comp(n, axes, point):
        ans = pauli_ansatz(axes=axes)
        op = get_op(n)
        logging.info(f'Computing gradient on {n} qubits')
        grad = get_gradient_fd(ans=ans, op=op, point=point)
        logging.info(f'Finished computing gradient on {n} qubits')
        return {
            'n': n,
            'l': len(axes),
            'grad': grad.real,
            'ans': 'rand'
        }

    logging.info('Defining bags')
    input_bag = db.from_sequence(inputs)
    bag = input_bag.map(lambda kwargs: grad_comp(**kwargs))
    logging.info('Defining dataframe')
    df = bag.to_dataframe()
    return df

def gen_mcp_circs(
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
                inputs.append({'n': n, 'l': l, 'point': point, 'axes': axes})
    logging.info(f'Defined {len(inputs)} experiments')

    def grad_comp(n, axes, point, l):
        ans = pauli_ansatz(axes)
        op = get_op(n)

        logging.info(f'Computing gradient on {n} qubits')
        grad = get_gradient_fd(ans=ans, op=op, point=point)
        logging.info(f'Finished computing gradient on {n} qubits')
        return {
            'n': n,
            'l': l,
            'grad': grad.real,
            'ans': 'mcp'
        }

    logging.info('Defining bags')
    input_bag = db.from_sequence(inputs)
    bag = input_bag.map(lambda kwargs: grad_comp(**kwargs))
    logging.info('Defining dataframe')
    df = bag.to_dataframe()
    return df


def gen_spc_circs(
    qubits_range: List[int],
    num_samples: int,
    time_reversal_symmetry: bool = True
):
    inputs = []

    logging.info('Defining experiments for SPC circs')

    for n in qubits_range:
        for m in range(n):
            for _ in range(num_samples):
                theta_pars = np.random.uniform(-np.pi, +np.pi, size=comb(n, m) - 1)
                if time_reversal_symmetry:
                    phi_pars = np.zeros(len(theta_pars))
                else:
                    phi_pars = np.random.uniform(-np.pi, +np.pi, size=comb(n, m) - 1)
                point = np.concatenate((theta_pars, phi_pars))
                inputs.append({'n': n, 'm': m, 'point': point})
    logging.info(f'Defined {len(inputs)} experiments')

    # Shuffle so jobs are more homogeneous
    logging.info('Shuffling experiments...')
    shuffle(inputs)

    def grad_comp(n, m, point):
        ans = spc_ansatz(num_qubits=n, num_particles=m)
        op = get_op(n)
        logging.info(f'Computing gradient on {n} qubits')
        if time_reversal_symmetry:
            grad_pars = range(comb(n, m) - 1)
        else:
            grad_pars = range(2*(comb(n, m) - 1))
        grad = get_gradient_fd(ans=ans, op=op, point=point, grad_pars=grad_pars)
        logging.info(f'Finished computing gradient on {n} qubits')
        return {
            'n': n,
            'm': m,
            'time_rev_sym': time_reversal_symmetry,
            'grad': grad.real,
            'ans': 'spc'
        }

    logging.info('Defining bags')
    input_bag = db.from_sequence(inputs)
    bag = input_bag.map(lambda kwargs: grad_comp(**kwargs))
    logging.info('Defining dataframe')
    df = bag.to_dataframe()
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

    num_samples = 50
    qubits = [4, 6]

    #df = gen_random_circs(
    #    qubits_range=qubits,
    #    num_samples=num_samples,
    #    layers=list(range(10, 100, 10))
    #)

    df = gen_spc_circs(
        qubits_range=qubits,
        num_samples=num_samples,
        time_reversal_symmetry=True
    )

    #df = gen_mcp_circs(
    #    qubits_range=qubits,
    #    num_samples=num_samples,
    #    depth_range=[10, 20]
    #)

    df = df.explode('grad')
    df['grad'] = df['grad'].astype(float)
    logging.info('Defining pivot...')
    pivot = df.groupby(['n', 'm']).agg({'grad': ['mean', 'std']})
    logging.info('Dumping to file...')
    pivot.to_csv('data/spc/output_*.csv')
    logging.info('Dumped to file...')
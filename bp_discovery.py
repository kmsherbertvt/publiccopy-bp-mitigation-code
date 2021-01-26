import numpy as np
from math import comb
from dask.distributed import Client, progress, wait
from typing import List
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
    pauli_ansatz
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


if __name__ == '__main__':
    logging.info('Starting dask cluster/client...')
    t_o = time.time()
    np.random.seed(42)
    cluster = SLURMCluster(extra=["--lifetime-stagger", "2m"])
    #cluster.adapt(minimum_jobs=3, maximum_jobs=20)

    logging.info('Scaling cluster...')
    cluster.scale(jobs=20)
    client = Client(cluster)
    client.upload_file('simulator.py')

    # Barren plateaus for MCPs
    max_num_qubits = 6
    qubits_range = range(4, max_num_qubits+2)[0:max_num_qubits:2]
    num_samples = 50
    #layers = list(range(10, 100, 10)) + list(range(100,1000,100))
    layers = range(10, 200, 10)

    inputs = []

    logging.info('Defining experiments...')
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
            'grad': grad.real
        }

    logging.info('Defining bags')
    input_bag = db.from_sequence(inputs)
    bag = input_bag.map(lambda kwargs: grad_comp(**kwargs))
    logging.info('Defining dataframe')
    df = bag.to_dataframe()
    df = df.explode('grad')
    df['grad'] = df['grad'].astype(float)
    logging.info('Defining pivot...')
    pivot = df.groupby(['l', 'n']).agg({'grad': ['mean', 'std']})
    logging.info('Dumping to file...')
    pivot.to_csv('data/rand/bp_rand_sum_*.csv')
    logging.info('Dumped to file...')
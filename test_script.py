import numpy as np
from math import comb
from dask.distributed import Client, progress, wait
from dask_jobqueue import PBSCluster
import dask
import dask.bag as db
import pandas as pd
import time

from simulator import (
    Array,
    pauli_str,
    get_gradient_fd,
    spc_ansatz,
    pauli_ansatz
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
    np.random.seed(42)
    cluster = PBSCluster(n_workers=24)
    cluster.scale(n=48)
    client = Client(cluster)
    client.upload_file('simulator.py')

    # Barren plateaus for MCPs
    max_num_qubits = 8
    qubits_range = range(4, max_num_qubits+2)[0:max_num_qubits:2]
    num_samples = 20
    #layers = list(range(10, 100, 10)) + list(range(100,1000,100))
    layers = range(10, 100, 10)

    inputs = []

    for n in qubits_range:
        for l in layers:
            axes = [
                np.random.choice([0, 1, 2, 3], size=n)
                for _ in range(l)
            ]
            for i in range(num_samples):
                point = np.random.uniform(-np.pi, +np.pi, size=l)
                inputs.append({'n': n, 'axes': axes, 'point': point})
    
    input_bag = db.from_sequence(inputs)

    def grad_comp(n, axes, point):
        ans = pauli_ansatz(axes=axes)
        op = get_op(n)
        grad = get_gradient_fd(ans=ans, op=op, point=point)
        return {
            'n': n,
            'l': len(axes),
            'grad': grad.real
        }

    bag = input_bag.map(lambda kwargs: grad_comp(**kwargs))

    df = bag.to_dataframe()
    df = df.explode('grad')
    
    time = time.time()
    df.to_csv(f'data/bp_spc_*.csv')
import numpy as np
from math import comb
from dask.distributed import Client, progress, wait
from dask_jobqueue import PBSCluster
import dask
import pandas as pd
import time

from simulator import (
    Array,
    pauli_str,
    get_gradient_fd,
    spc_ansatz
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
    max_num_qubits = 4
    num_samples = 2
    qubits_range = range(4, max_num_qubits+2)[0:max_num_qubits:2]

    lazy_data = []

    @dask.delayed
    def get_grad_sample(n, m, ans, op, point):
        grad = get_gradient_fd(ans=ans, op=op, point=point)
        return {
            'n': n,
            'm': m,
            'grad': list(grad.real)
        }

    for n in qubits_range:
        op = get_op(n)
        for m in range(2, n):
            ans = spc_ansatz(n, m)
            num_pars = 2*(comb(n, m) - 1)
            for i in range(num_samples):
                point = list(np.random.uniform(-np.pi, +np.pi, size=num_pars//2))
                point = point + [0.0]*(num_pars//2)

                lazy_data.append(get_grad_sample(
                    n=n, m=m, ans=ans, op=op, point=point
                ))
            
    cluster = PBSCluster()
    cluster.scale(1)
    client = Client(cluster)
    client.upload_file('simulator.py')

    futures = dask.persist(*lazy_data)
    results = dask.compute(*futures)
    df = pd.DataFrame(results)
    time = time.time()
    df.to_csv(f'bp_spc_{time}.csv')
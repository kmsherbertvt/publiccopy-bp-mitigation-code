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
    ParameterStopper
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
    qaoa_ansatz,
    make_connectivity_pool,
    plus_state
)

logging.basicConfig(
    filename='adapt_benchmark.log',
    level=logging.INFO
)


def adapt_run(d: Dict) -> Dict:
    np.random.seed(d['seed'])
    if d['pool_name'] == 'path':
        graph = nx.path_graph(d['n'])
        pool = make_connectivity_pool(graph)
    else:
        raise NotImplementedError
    
    def optimizer(*args, **kwargs):
        return minimize(*args, method='Nelder-Mead', **kwargs)
    
    ham = np.diag(np.random.uniform(size=2**d['n']))
    e_min = np.min(np.diagonal(ham))
    e_max = np.max(np.diagonal(ham))
    gap = e_max - e_min

    callbacks = [ParameterStopper(10)]
    result = adapt(
        ham=ham,
        pool=pool,
        optimizer=optimizer,
        callbacks=callbacks,
        initial_state=plus_state(d['n']),
        new_parameter=1e-2
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
        'max_grads': [res['max_grad'] for res in result.step_history],
        'halt_reason': result.halt_reason
    }
    del result
    del ham
    del graph
    del pool
    
    return d_out


def submit_futures(l: List[dict], client: Client) -> List[Future]:
    logging.info('Submitting futures')
    futures = client.map(adapt_run, l, retries=1)
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
    client.upload_file('adapt.py')

    # Define experiments
    num_samples = 100
    qubits = [3, 4, 5, 6]
    pool_names = ['path']
    seed = 1

    experiments = []
    for n in reversed(qubits):
        for pool_name in pool_names:
            for _ in range(num_samples):
                experiments.append({
                    'pool_name': pool_name,
                    'n': n,
                    'seed': seed
                })
                seed += 1
    #shuffle(experiments)

    futures = submit_futures(experiments, client=client)
    res_list = []
    failed_futures = 0
    for future in as_completed(futures, with_results=False):
        if future.status == 'error':
            logging.info(f'Future failed: {future.exception()}')
            logging.info(f'Future failed with traceback: {future.traceback()}')
            failed_futures += 1
        elif future.status == 'lost':
            logging.info(f'Future lost: {future}')
        elif future.status == 'cancelled':
            logging.info(f'Future cancelled: {future}')
        elif future.status == 'finished':
            try:
                res_list.append(future.result())
            except:
                logging.info('Some weird error...')
        del future
    logging.info(f'Completed running futures, {failed_futures} failed')
    logging.info('Making dataframe...')
    df = pd.DataFrame(res_list)
    logging.info('Writing to disk')
    df.to_csv('data/'+filename_format(__file__, 'test'))
    logging.info('Wrote to disk, exiting!')
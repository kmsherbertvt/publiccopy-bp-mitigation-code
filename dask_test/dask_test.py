import dask
from dask.distributed import Client, progress
from dask_jobqueue import PBSCluster
import time
import numpy as np


@dask.delayed
def add_slow(x, y) -> int:
    time.sleep(np.random.uniform())
    return x+y


if __name__ == '__main__':
    cluster = PBSCluster()
    cluster.scale(2)
    client = Client(cluster)

    results = []
    for i in range(100):
        x = np.random.uniform()
        y = np.random.uniform()
        results.append(add_slow(x, y))
    
    mean = np.mean(results)
    print(client.compute(mean))
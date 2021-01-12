import dask
import dask.bag as db
from dask.distributed import Client, progress
from dask_jobqueue import SLURMCluster
import time
import numpy as np


if __name__ == '__main__':
    cluster = SLURMCluster(
        n_workers=10
        )
    cluster.scale(1)

    with open('dask_job_script.sh', 'w') as f:
        f.write(cluster.job_script())

    client = Client(cluster)

    b1 = db.from_sequence(range(1000))
    b2 = b1.map(lambda x: x+1)

    res = b2.mean()
    print(res)

    print(res.compute())

from contextlib import contextmanager

import multiprocessing as multi


@contextmanager
def pooler(*args, **kwargs):

    pool = multi.Pool(*args, **kwargs)
    yield pool
    pool.close()
    pool.join()
    pool.terminate()

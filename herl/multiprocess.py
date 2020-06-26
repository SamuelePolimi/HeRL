import numpy as np

from joblib import Parallel, delayed
from typing import Iterable, Callable

from herl.utils import Printable


class MultiProcess(Printable):

    def __init__(self, n_process=20, random_seed=True):
        self._n_process = n_process
        self._use_random_seed = random_seed

    def compute(self, process: Callable, params=10, dict_args=None) -> Iterable:

        if type(params) is int:
            args_params = [[] for _ in range(params)]
        else:
            args_params = params

        if dict_args is None:
            kwargs_params = [{} for _ in range(params)]
        else:
            kwargs_params = dict_args

        def rnd_process(*args, **kwargs):
            if self._use_random_seed:
                np.random.seed()
            return process(*args, **kwargs)

        return Parallel(n_jobs=self._n_process)(delayed(rnd_process)(*a, **k)
                                                for a, k in zip(args_params, kwargs_params))
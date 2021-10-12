import numpy as np
import torch
from typing import List, Callable

from herl.utils import _one_hot
from herl.rl_interface import RLEnvironmentDescriptor


class Features:

    def __init__(self):
        pass

class CriticFeatures:

    def __init__(self, feature_constructor: Callable, n_features:List[int], env_descriptor: RLEnvironmentDescriptor, **kwargs):
        dims = []
        for i in range(env_descriptor.state_space.low.shape[0]):
            dims.append((env_descriptor.state_space.low[i], env_descriptor.state_space.high[i]))
        for i in range(env_descriptor.action_space.low.shape[0]):
            dims.append((env_descriptor.action_space.low[i], env_descriptor.action_space.high[i]))
        self._features = feature_constructor(dims, n_features, **kwargs)

    def __call__(self, s, a, differentiable=False):
        ret = self._features(np.concatenate([s.detach().numpy(), a.detach().numpy()], axis=-1))
        return torch.tensor(ret)

class TileCoding:

    def __init__(self, ranges: List[np.ndarray], n_coding: List[int]):
        self._ranges = ranges
        self._n_coding = n_coding
        self._M = [np.linspace(range[0], range[1], n, endpoint=False) for range, n in zip(ranges, n_coding)]

    def _one_number(self, x):
        prod = np.cumprod([1] + self._n_coding)
        tot = prod[-1]
        if len(x.shape)==1:
            return _one_hot(np.sum(x * prod[:-1]), tot)
        else:
            return _one_hot(np.sum(x * prod[:-1], axis=1), tot)

    def __call__(self, x):
        def arg_where(m, v):
            where = np.argwhere((np.array([m]).T <= v).T)
            indexes = np.argwhere(where[:, 0] - np.concatenate([where[1:, 0], np.array([np.inf])]) < 0).ravel()
            return where[indexes, 1]
        if len(x.shape) == 1:
            ret = np.array([np.max(np.argwhere(self._M[i] <= x_i)) for i, x_i in enumerate(x)])
            return self._one_number(ret)
        else:
            try:
                ret = np.array([arg_where(self._M[i], x_i) for i, x_i in enumerate(x.T)]).T
            except:
                print("whatever")
            return self._one_number(ret)

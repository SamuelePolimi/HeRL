import numpy as np
import multiprocessing as mp
from multiprocessing.pool import ThreadPool as Pool
from typing import Callable, Tuple, List, Union

from herl.rl_interface import RLTask, RLAgent, Critic, PolicyGradient, Actor, Online
from herl.utils import Printable


class BaseAnalyzer(Printable):

    def __init__(self, verbose=True, plot=True):
        Printable.__init__(self, "Analyzer", verbose, plot)


def montecarlo_estimate(task, state=None, action=None, policy=None, abs_confidence=0.1):
    """

    :param task: The task we want to estimate
    :type task: RLTask
    :param state: a particular state
    :param action: if you want to estimate the value, set to None.
    :param policy: the policy you want to estimate
    :type policy: RLAgent
    :param gamma_accuracy:
    :param abs_confidence:
    :return:
    """
    copy_task = task.copy()
    env = copy_task.environment

    if (state is not None or env.is_init_deterministic()) and policy.is_deterministic() and env.is_deterministic():
        ret = copy_task.episode(policy, state, action)
        return ret
    else:
        j_list = []
        current_std = np.inf
        while current_std > abs_confidence or len(j_list) <= 1:
            j_list.append(copy_task.episode(policy, state, action))
            current_std = 1.96 * np.std(j_list) / len(j_list)
        return np.mean(j_list)


class MCAnalyzer(Critic, PolicyGradient, Online):
    """
    This class perform an estimation of the critic and the gradient using Monte-Carlo sampling.
    For this, a settable environment is needed.
    """

    def __init__(self, rl_task, policy):
        name = "MC"
        Actor.__init__(self, name, policy)
        Online.__init__(self, name, rl_task)

    def get_Q(self, state, action, abs_confidence=0.1):
        return montecarlo_estimate(self._task, state, action, self.policy, abs_confidence)

    def get_V(self, state, abs_confidence=0.1):
        if len(state.shape) == 1:
            return montecarlo_estimate(self._task, state, policy=self.policy, abs_confidence=abs_confidence)
        else:
            pool = Pool(mp.cpu_count())
            f = lambda x: montecarlo_estimate(self._task, x, policy=self.policy, abs_confidence=abs_confidence)
            v = pool.map(f, state)
            return np.array(v)

    def get_return(self, abs_confidence=0.1):
        return np.asscalar(montecarlo_estimate(self._task, policy=self.policy, abs_confidence=abs_confidence))

    def get_gradient(self, delta=1E-3, abs_confidence=0.1):
        params = self.policy.get_parameters().copy()
        j_ref = montecarlo_estimate(self._task, policy=self.policy, abs_confidence=abs_confidence)
        grad = np.zeros_like(params)
        for i in range(params.shape[0]):
            new_params = params.copy()
            new_params[i] = params[i] + delta
            self.policy.set_parameters(new_params)
            j_delta = montecarlo_estimate(self._task, policy=self.policy, abs_confidence=abs_confidence)
            grad[i] = (j_delta - j_ref)/delta
            self.policy.set_parameters(params)
        return grad


def bias_variance_estimate(ground_thruth: Union[float, np.ndarray], estimator_sampler: Callable,
                           abs_confidence: float = 1E-1, min_samples: int = 10, max_sample: int = 20)\
        -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], float]:
    """

    :param ground_thruth:
    :param estimator_sampler:
    :param confidence:
    :rtype:
    :return:
    """

    estimate_list = []
    current_std = np.inf
    while (current_std > abs_confidence or len(estimate_list) <= min_samples) and len(estimate_list)<=max_sample:
        estimate_list.append(estimator_sampler())
        current_std = 1.96 * np.std(estimate_list) / np.sqrt(len(estimate_list))

    mean_estimate = np.mean(estimate_list, axis=0)
    variance_list = []

    for estimate in estimate_list:
        variance_list.append((mean_estimate-estimate)**2)

    variance_estimate = np.mean(variance_list, axis=0)
    bias_estimate = mean_estimate - ground_thruth

    return bias_estimate, variance_estimate, estimate_list, current_std


def gradient_direction(ground_truth: np.ndarray, gradients: np.ndarray) -> np.ndarray:
    """
    Receives two matrixes of arrays. each row contains a vector.
    The method return a 1D-array of angles.
    :param ground_truth: (n x d), it contains n vectors of dimension d to be compared
    :param gradients: (n x d), it contains n vectors of dimension d to be compared
    :return: a vector of n angles between 0 and pi.
    """

    norm = 1/(np.linalg.norm(ground_truth, axis=1)*np.linalg.norm(gradients, axis=1))
    cos_x = norm * np.einsum('ij,ji->i', ground_truth, gradients.T)
    return np.arccos(cos_x)

import numpy as np

from typing import Callable, Tuple, List, Union

import torch
from scipy.stats import chi2
import time
import matplotlib.pyplot as plt

from herl.rl_interface import RLTask, RLAgent, Critic, PolicyGradient, Actor, Online, RLParametricModel
from herl.utils import Printable
from herl.multiprocess import MultiProcess
from herl.classic_envs import MDP
from herl.actor import TabularPolicy



class BaseAnalyzer(Printable):

    def __init__(self, verbose=True, plot=True):
        Printable.__init__(self, "Analyzer", verbose, plot)


class MCEstimate:

    def __init__(self, task, min_samples=1, max_samples=100, absolute_confidence=1E-1, gamma_precision=0):
        self.task = task
        self.min_samples = min_samples
        self.max_samples = max_samples
        self.gamma_precision = gamma_precision
        self.absolute_donfidence = absolute_confidence

    def estimate(self, state=None, action=None, policy=None):
        return montecarlo_estimate(self.task, state, action, policy, abs_confidence=self.absolute_donfidence,
                                   min_samples=self.min_samples,
                                   max_samples=self.max_samples)


def montecarlo_estimate(task, state=None, action=None, policy=None, abs_confidence=0.1,
                        min_samples=1, max_samples=100):
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
        while (current_std > abs_confidence or len(j_list) <= min_samples) and len(j_list) < max_samples:
            j_list.append(copy_task.episode(policy, state, action))
            current_std = 1.96 * np.std(j_list) / np.sqrt(len(j_list))
        print("MC Estimate. Samples: %d, Confidence: %f" % (len(j_list), current_std))
        return np.mean(j_list)


class MCAnalyzer(Critic, PolicyGradient, Online):
    """
    This class perform an estimation of the critic and the gradient using Monte-Carlo sampling.
    For this, a settable environment is needed.
    """
    # TODO: use the mc_estimators
    def __init__(self, rl_task: RLTask, policy: Union[RLAgent, RLParametricModel],
                 q_mc_estimator=None, v_mc_estimator=None, return_mc_estimator=None, gradient_mc_estimator=None,
                 delta_gradient=1E-3):
        name = "MC"
        Actor.__init__(self, name, policy)  # TODO is it correct??
        Online.__init__(self, name, rl_task)
        not_none = lambda estimator: estimator if estimator is not None else MCEstimate(rl_task)
        self.q_mc_estimator = not_none(q_mc_estimator)
        self.v_mc_estimator = not_none(v_mc_estimator)
        self.return_mc_estimator = not_none(return_mc_estimator)
        self.gradient_mc_estimator = not_none(gradient_mc_estimator)
        self.delta_gradient = not_none(delta_gradient)

    def get_Q(self, state, action):
        return self.q_mc_estimator.estimate(state, action, self.policy)

    def get_V(self, state):
        if len(state.shape) == 1:
            return self.v_mc_estimator.estimate(state, policy=self.policy)
        else:
            pool = ProcessPool(mp.cpu_count())
            f = lambda x: self.v_mc_estimator.estimate(x, policy=self.policy)
            v = pool.map(f, state)
            return np.array(v)

    def get_return(self):
        return np.asscalar(self.return_mc_estimator.estimate(policy=self.policy))

    def get_gradient(self):
        params = self.policy.get_parameters().copy()
        j_ref = self.gradient_mc_estimator.estimate(policy=self.policy)
        grad = np.zeros_like(params)
        for i in range(params.shape[0]):
            new_params = params.copy()
            new_params[i] = params[i] + self.delta_gradient
            self.policy.set_parameters(new_params)
            j_delta = self.gradient_mc_estimator.estimate(policy=self.policy)
            grad[i] = (j_delta - j_ref)/self.delta_gradient
            self.policy.set_parameters(params)
        return grad


class OnlineEstimate:

    def __init__(self):
        """

        """
        self._count = 0
        self._mean = 0.
        self._m2 = 0.
        self._variance = 0.

    def notify(self, newValue):
        self._count += 1
        delta = newValue - self._mean
        self._mean += delta / self._count
        delta2 = newValue - self._mean
        self._m2 += delta * delta2
        if self._count < 2:
            self._variance = np.nan
            return np.inf
        else:
            self._variance = self._m2 / (self._count - 1)
            return 1.96 * np.sqrt(self._variance) / np.sqrt(self._count)

    def get_mean(self):
        return self._mean

    def get_variance(self):
        return self._variance

    def get_mean_confidence_interval_95(self, stats='normal'):
        if stats == 'normal':
            return 1.96 * np.sqrt(self._variance) / np.sqrt(self._count)
        if stats == 'square':
            return chi2.ppf(0.025, self._count) / self._count

    def get_variance_confidence_interval_95(self):
        return (self._count - 1) * self._variance / chi2.ppf(0.025, self._count-1) - self._variance \
            , (self._count - 1) * self._variance / chi2.ppf(0.975, self._count-1) - self._variance

    def get_count(self):
        return self._count

    def __str__(self):
        conf_m, conf_p = self.get_variance_confidence_interval_95()
        return """Estimated average: %s +- %s (confidence 95%%),
        Estimated variance: %s (+ %s - %s) (confidence 95%%)
        Number of samples %d
        """ % (self.get_mean(), self.get_mean_confidence_interval_95(), self.get_variance(), conf_m, conf_p, self.get_count())


def bias_variance_estimate(ground_thruth: Union[float, np.ndarray], estimator_sampler: Callable,
                           abs_confidence: float = 1E-1, n_inner_loop_samples: int = 10, min_samples=2, max_samples: int = 20)\
        -> Tuple[OnlineEstimate, OnlineEstimate, OnlineEstimate]:
    """

    :param ground_thruth:
    :param estimator_sampler:
    :param confidence:
    :rtype:
    :return:
    """
    variance_estimate = OnlineEstimate()
    bias_estimate = OnlineEstimate()
    mse_estimate = OnlineEstimate()
    mp = MultiProcess()
    while True:
        #estimate_list = thread.map(lambda x: estimator_sampler(), range(n_inner_loop_samples))
        estimate_list = mp.compute(estimator_sampler, n_inner_loop_samples)
        for e in estimate_list:
            variance_estimate.notify(e)
            mse_estimate.notify((e - ground_thruth)**2)
            bias_estimate.notify(e-ground_thruth)

        if bias_estimate.get_count() < min_samples:
            continue

        if np.sum(mse_estimate.get_mean_confidence_interval_95()) < abs_confidence:
            break

        if bias_estimate.get_count() >= max_samples:
            break

    return bias_estimate, variance_estimate, mse_estimate


def bias_variance_estimate_backup(ground_thruth: Union[float, np.ndarray], estimator_sampler: Callable,
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
    The method return a 1D-array of angles between 0 and pi.
    :param ground_truth: (n x d), it contains n vectors of dimension d to be compared
    :param gradients: (n x d), it contains n vectors of dimension d to be compared
    :return: a vector of n angles between 0 and pi.
    """
    ground_truth_copy = ground_truth
    if len(ground_truth.shape) < 2:
        ground_truth_copy = np.repeat([ground_truth], gradients.shape[1])
    norm = 1/(np.linalg.norm(ground_truth_copy, axis=1)*np.linalg.norm(gradients, axis=1))
    cos_x = norm * np.einsum('ij,ji->i', ground_truth_copy, gradients.T)
    return np.arccos(cos_x)


def gradient_2d_full_direction(ground_truth: np.ndarray, gradients: np.ndarray) -> np.ndarray:
    """
    Receives tro matrixe of arrays. each row contains a 2d vector.
    The method returns a 1D-array of angles between -pi and pi.
    :param ground_truth:
    :param gradients:
    :return:
    """
    ground_truth_angle = np.angle(ground_truth[0] + ground_truth[1]* 1j)
    gradients_angles = np.angle(gradients[:, 0] + gradients[:, 1]* 1j) - ground_truth_angle
    gradients_angles = np.vectorize(norm_angle)(gradients_angles)
    residual_mean = np.mean(gradients[:, 0] + gradients[:, 1]* 1j, axis=0)
    return gradients_angles, norm_angle(np.angle(residual_mean) - ground_truth_angle)


def norm_angle(angle):
    if angle > np.pi:
        return norm_angle(angle - 2*np.pi)
    if angle < -np.pi:
        return norm_angle(angle + 2*np.pi)
    return angle

class MDPAnalyzer:

    def __init__(self, task:RLTask, policy:TabularPolicy):
        self._task = task
        if type(self._task.environment) is not MDP:
            raise("task.environemnt should be an MDP!")
        self._mdp = task.environment # type: MDP
        self._policy = policy

        self._M = self._policy.tabular
        self._P = self._mdp.get_transition_matrix()
        self._r = self._mdp.get_reward_matrix()
        self._mu_0 = self._mdp.get_initial_state_probability()

        self._n_states = len(self._mdp.get_states())
        self._n_actions = len(self._mdp.get_actions())

        self._gamma = task.gamma

    def get_P_policy(self):
        P = torch.zeros((self._n_states, self._n_states))
        for s in self._mdp.get_states():
            pi = self._M[s]

            T = 0.
            for a, p_a in enumerate(pi):
                T +=  torch.tensor(self._P[a, s]) * p_a

            P[s, :] = T

        return P

    def get_r_policy(self):
        r = torch.Tensor((self._n_states))

        for s in self._mdp.get_states():
            r[s] = torch.inner(torch.tensor(self._r[:, s]), self._M[s, :])

        return r

    def get_v(self):
        P = self.get_P_policy()
        r = self.get_r_policy()
        return torch.linalg.inv(torch.eye(P.shape[0]) - self._gamma*P) @ r

    def get_return(self):
        return torch.inner(torch.tensor(self._mu_0, dtype=torch.float64), self.get_v().type(dtype=torch.float64))

    def get_policy_gradient(self):
        j = self.get_return()
        j.backward()
        return self._policy.get_gradient()


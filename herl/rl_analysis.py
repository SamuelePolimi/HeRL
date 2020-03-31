import numpy as np
import multiprocessing as mp

from herl.rl_interface import RLTask, RLAgent, Critic, PolicyGradient, Actor, Online
from herl.dataset import Dataset, Domain
from herl.solver import RLCollector


class Analyser:

    def __init__(self, verbose=True, plot=True):
        self.verbose = verbose
        self.plot = plot

    def print(self, string):
        if self.verbose:
            print("Analyzer: %s" % string)


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
        return copy_task.episode(policy, state, action)
    else:
        j_list = []
        current_std = np.inf
        while current_std > abs_confidence or len(j_list) <= 1:
            j_list.append(copy_task.episode(policy, state, action))
            current_std = 1.96 * np.std(j_list) / len(j_list)
        return np.mean(j_list)


class MCAnalyzer(Critic, Actor, PolicyGradient, Online):
    """
    This class perform an estimation of the critic and the gradient using Monte-Carlo sampling.
    For this, a settable environment is needed.
    """

    def __init__(self, rl_task, policy):
        name = "Monte-Carlo Analyzer"
        Actor.__init__(self, name, policy)
        Online.__init__(self, name, rl_task)

    def get_Q(self, state, action, abs_confidence=0.1):
        return montecarlo_estimate(self._task, state, action, self.policy, abs_confidence)

    def get_V(self, state, abs_confidence=0.1):
        return montecarlo_estimate(self._task, state, policy=self.policy, abs_confidence=abs_confidence)

    def get_return(self, abs_confidence=0.1):
        return montecarlo_estimate(self._task, policy=self.policy, abs_confidence=abs_confidence)

    def get_gradient(self, delta=1E-2, abs_confidence=0.1):
        params = self.policy.get_parameters().copy()
        j_ref = montecarlo_estimate(self._task, policy=self.policy, abs_confidence=abs_confidence)
        grad = np.zeros_like(params)
        for i in range(params.shape[0]):
            new_params = params.copy()
            new_params[i] = params[i] + delta
            self.policy.set_parameters(new_params)
            j_delta = montecarlo_estimate(self._task, policy=self.policy, abs_confidence=abs_confidence)
            grad[i] = (j_ref - j_delta)/delta
            self.policy.set_parameters(params)
        return grad
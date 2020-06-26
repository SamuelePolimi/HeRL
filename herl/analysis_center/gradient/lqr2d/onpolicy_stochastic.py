"""
This analyser provides a neural network randomly initialized, and evaluate the prediction of the algorithm under
different tests, plotting the value function, the return under different datasets.
"""
import numpy as np
import os

from herl.actor import NeuralNetworkPolicy, LinearGaussianPolicy
from herl.rl_interface import RLTask, DeterministicState
from herl.classic_envs import LQR
from herl.analysis_center.gradient.gradient_analyzer import GradientAnalyzer
from herl.analysis_center.lqr_solver import check_statbility, LQRAnalyzer
from herl.dataset import Dataset, Domain, Variable
from herl.solver import RLCollector
from herl.utils import Printable

lqr = LQR(A=np.array([[1.2, 0.],
                      [0, 1.1]]),
          B=np.array([[1., 0.],
                      [0., 1.]]),
          R=np.array([[1., 0.],
                      [0., 1.]]),
          Q=np.array([[0.1, 0.],
                      [0., 0.1]]),
          initial_state=np.array([-1., -1.]),
          state_box=np.array([2., 2.]),
          action_box=np.array([2., 2.]))

covariance = 0.1 * np.eye(2)

initial_state = DeterministicState(np.array([-1., -1.]))
task = RLTask(lqr, initial_state, gamma=0.9, max_episode_length=200)

parameter_distribution = lambda: np.random.multivariate_normal(np.array([-1.2, -1.1]), 0.00001 * np.eye(2))


def _get_path(filename):
    full_path = os.path.realpath(__file__)
    path, _ = os.path.split(full_path)
    return path + "/" + filename


def _new_network() -> NeuralNetworkPolicy:
    ret = LinearGaussianPolicy(2, 2, covariance=covariance, diagonal=True)
    parameters = parameter_distribution()
    while not check_statbility(lqr._A, lqr._B, np.diag(parameters)):
        parameters = parameter_distribution()
    ret.set_parameters(parameters)
    return ret


for _ in range(100):
    _new_network()


def _new_parameters(n_parameters=100):
    return np.array([_new_network().get_parameters() for _ in range(n_parameters)])


def _load_data():
    n_params = _new_network().get_parameters().shape[0]
    dataset = Dataset.load(_get_path("/data/on_s_gradients.npz"),
                 Domain(Variable("parameters", n_params),
                        Variable("gradients", n_params))
                 )
    data = dataset.get_full()
    policies = []
    for params in data['parameters']:
        policy = _new_network()
        policy.set_parameters(params)
        policies.append(policy)
    return policies, data['parameters'], data['gradients']


class Reset(Printable):

    def __init__(self):
        Printable.__init__(self, "Reset")
        self.parameters = None
        self.gradients = None

    def reset(self):
        """
        By calling this method, you overwrite the current policy, and all the results will be computed and
        saved on disk.
        :return:
        """
        self.reset_policies()
        self.reset_gredients()
        self.save()

    def reset_policies(self):
        self.print("Policy reset.")
        self.parameters = _new_parameters(100)

    def reset_gredients(self):
        self.print("Gradient reset.")
        gradients = []
        progress = self.get_progress_bar("Gradient Computation")
        for parameter in self.parameters:
            progress.notify()
            policy = _new_network()
            policy.set_parameters(parameter)
            analyzer = LQRAnalyzer(task, policy)
            gradient = analyzer.get_gradient()
            gradients.append(gradient)
        self.gradients = np.array(gradients)

    def save(self):
        dataset = Dataset(Domain(Variable("parameters", self.parameters.shape[1]),
                       Variable("gradients", self.gradients.shape[1])), n_max_row=self.gradients.shape[0])
        dataset.notify_batch(parameters=self.parameters, gradients=self.gradients)
        dataset.save(_get_path("/data/on_s_gradients.npz"))


# reset = Reset()
# reset.reset()


class OnPolicyStochasticGradientAnalyzer(GradientAnalyzer):

    def __init__(self, **algorithm_constructors):
        """

        :param algorithm_constructors:
        """
        GradientAnalyzer.__init__(self, task, **algorithm_constructors)
        self.print("Loading Data.")
        self.policies, self.parameters, self.gradients = _load_data()
        self.print("Data Loaded.")
        self.print("""OnPolicyDeterministicGradientAnalyzer.
        The purpose of this analyzer is to evaluate the gradient estimation of a deterministic 2D pendulum.""")

    def all_analysis(self):
        pass

    def visualize_gradient_estimates(self, n_policies=100, n_samples=2000, **graphic_args):
        def get_dataset(agent):
            dataset = self.task.get_empty_dataset(n_max_row=int(n_samples))
            collector = RLCollector(dataset, task.copy(), agent)
            collector.collect_samples(n_samples)
            return dataset
        self.base_off_policy_gradient_direction_estimates(self.policies[:n_policies], self.gradients, get_dataset)

    def visalize_bias_variance_estimates(self, confidence=0.5, n_policies=100, n_samples=None):
        # keep
        if n_samples is None:
            n_samples = [100, 200, 500, 1000, 2000, 3000, 4000]

        for ground_truth, policy in zip(self.gradients[:n_policies], self.policies[:n_policies]):
            def get_dataset(n):
                dataset = self.task.get_empty_dataset(n_max_row=int(n))
                collector = RLCollector(dataset, task.copy(), policy)
                collector.collect_samples(n)
                return dataset
            return self.base_bias_variance_gradient(ground_truth, policy, get_dataset, parameters=n_samples,
                                                  confidence=confidence,
                                                  inner_samples=1,
                                                  min_samples=2,
                                                  max_samples=100)

    def visualize_parametric_gradient_estimates(self, n_policies=100, n_samples=None):
        # keep
        if n_samples is None:
            n_samples = [2400, 2200, 2000, 1800, 1600, 250]

        def get_dataset(agent, n):
            dataset = self.task.get_empty_dataset(n_max_row=int(n))
            collector = RLCollector(dataset, task.copy(), agent)
            collector.collect_samples(n)
            return dataset

        return self.base_gradient_direction_samples(self.policies[:n_policies], self.gradients[:n_policies], get_dataset,
                                                  n_samples)


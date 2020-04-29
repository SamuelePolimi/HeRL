"""
This analyser provides a neural network randomly initialized, and evaluate the prediction of the algorithm under
different tests, plotting the value function, the return under different datasets.
"""
import torch, numpy as np
import matplotlib.pyplot as plt
import os

from herl.actor import NeuralNetworkPolicy, UniformPolicy
from herl.rl_interface import RLTask, DeterministicState, StochasticState
from herl.classic_envs import Pendulum2D
from herl.rl_analysis import MCAnalyzer
from herl.rl_visualizer import GradientRowVisualizer, ReturnRowVisualizer, RowVisualizer
from herl.analysis_center.gradient.gradient_analyzer import GradientAnalyzer
from herl.dataset import Dataset, Domain, Variable
from herl.solver import RLCollector
from herl.utils import Printable

task = RLTask(Pendulum2D(), DeterministicState(np.array([np.pi, 0.])), gamma=0.95, max_episode_length=200)


def _get_path(filename):
    full_path = os.path.realpath(__file__)
    path, _ = os.path.split(full_path)
    return path + "/" + filename


def _new_network():
    return NeuralNetworkPolicy([50], [torch.relu], task, lambda x: 2 * torch.tanh(x))


def _new_parameters(n_parameters=100):
    return np.array([_new_network().get_parameters() for _ in range(n_parameters)])


def _load_data():
    n_params = _new_network().get_parameters().shape[0]
    dataset = Dataset.load(_get_path("pendulum_deterministic_data/data/gradients.npz"),
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
            analyzer = MCAnalyzer(task, policy)
            gradient = analyzer.get_gradient()
            gradients.append(gradient)
        self.gradients = np.array(gradients)

    def save(self):
        dataset = Dataset(Domain(Variable("parameters", self.parameters.shape[1]),
                       Variable("gradients", self.gradients.shape[1])), n_max_row=self.gradients.shape[0])
        dataset.notify_batch(parameters=self.parameters, gradients=self.gradients)
        dataset.save(_get_path("pendulum_deterministic_data/data/gradients.npz"))


# reset = Reset()
# reset.reset()


class Pendulum2DGradientAnalyzer(GradientAnalyzer):

    def __init__(self, *algorithm_constructors):
        """

        :param algorithm_constructors:
        """
        GradientAnalyzer.__init__(self, task, *algorithm_constructors)
        self.print("Loading Data.")
        self.policies, self.parameters, self.gradients = _load_data()
        self.print("Data Loaded.")
        self.print("""Pendulum2DGradientAnalyzer.
        The purpose of this analyzer is to evaluate the gradient estimation of a deterministic 2D pendulum.""")

    def all_analysis(self):
        pass

    def visualize_gradient_estimates_uniform_dataset(self, **graphic_args):
        dataset = task.environment.get_grid_dataset(states=np.array([25, 25]), actions=np.array([2]), step=True)
        # def get_dataset(n):
        #     random_start_task = RLTask(Pendulum2D(), gamma=0.95, max_episode_length=200)
        #     dataset = self.task.get_empty_dataset(n_max_row=int(n))
        #     collector = RLCollector(dataset, random_start_task, UniformPolicy(np.array([-2]), np.array([2])))
        #     collector.collect_samples(int(n))
        #     return dataset.train_ds
        self.visualize_off_policy_gradient_direction_estimates(self.policies, self.gradients, dataset)

    def onpolicy_bias_variance_estimates(self, confidence=5):
        self.print("The dataset is generated with rollout starting from the bottom position (as prescribed in the task)"
                   "and following the evaluation policy")

        for ground_truth, policy in zip(self.gradients, self.policies):
            def get_dataset(n):
                random_start_task = RLTask(Pendulum2D(),
                                       StochasticState(lambda: np.random.uniform(np.array([-np.pi, -8.]),
                                                       np.array([np.pi, 8.]))),
                                           gamma=0.95, max_episode_length=200)
                dataset = self.task.get_empty_dataset(n_max_row=int(n))
                collector = RLCollector(dataset, random_start_task, UniformPolicy(np.array([-2.]), np.array([2.])))
                collector.collect_samples(int(n))
                return dataset.train_ds
            self.visualize_bias_variance_gradient(ground_truth, policy, get_dataset, parameters=[200, 500, 1000, 2000],
                                                  confidence=confidence,
                                                  max_samples=100)
            self.show()

    def offpolicy_bias_variance_estimates(self, confidence=5):
        self.print("The dataset is generated with rollout starting from the bottom position (as prescribed in the task)"
                   "and following the evaluation policy")

        for ground_truth, policy in zip(self.gradients, self.policies):
            def get_dataset(n):
                random_start_task = RLTask(Pendulum2D(), gamma=0.95, max_episode_length=200)
                dataset = self.task.get_empty_dataset(n_max_row=int(n))
                collector = RLCollector(dataset, random_start_task, UniformPolicy(np.array([-2]), np.array([2])))
                collector.collect_samples(int(n))
                return dataset.train_ds
            self.visualize_bias_variance_gradient(ground_truth, policy, get_dataset, parameters=[100, 200, 500, 1000,
                                                                                                 2000, 3000, 4000],
                                                  confidence=confidence,
                                                  max_samples=100)
            self.show()

    def gradient_visualize(self, n_policies=100):
        def get_dataset(n):
            random_start_task = RLTask(Pendulum2D(),
                                       StochasticState(lambda: np.random.uniform(np.array([-np.pi, -8.]),
                                                       np.array([np.pi, 8.]))),
                                       gamma=0.95,
                                       max_episode_length=200)
            dataset = self.task.get_empty_dataset(n_max_row=int(n))
            collector = RLCollector(dataset, random_start_task, UniformPolicy(np.array([-2]), np.array([2])))
            collector.collect_samples(int(n))
            return dataset.train_ds
        self.visualize_gradient_direction_samples(self.policies[:n_policies], self.gradients[:n_policies], get_dataset,
                                                  [2000, 2750, 1500, 1250, 1000, 750, 500, 250])
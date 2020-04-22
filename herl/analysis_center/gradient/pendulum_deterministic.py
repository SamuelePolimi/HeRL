"""
This analyser provides a neural network randomly initialized, and evaluate the prediction of the algorithm under
different tests, plotting the value function, the return under different datasets.
"""
import torch, numpy as np
import matplotlib.pyplot as plt
import os

from herl.actor import NeuralNetworkPolicy
from herl.rl_interface import RLTask
from herl.classic_envs import Pendulum2D
from herl.rl_analysis import MCAnalyzer
from herl.rl_visualizer import GradientRowVisualizer, ReturnRowVisualizer, RowVisualizer
from herl.analysis_center.gradient.gradient_analyzer import GradientAnalyzer
from herl.dataset import Dataset, Domain, Variable
from herl.solver import RLCollector
from herl.utils import Printable

task = RLTask(Pendulum2D(np.array([np.pi, 0.])), gamma=0.95, max_episode_length=200)


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
        self.visualize_off_policy_gradient_direction_estimates(self.policies, self.gradients, dataset)

    def visualize_gradient_uniform_dataset(self, discretization=50, **graphic_args):

        dataset = task.environment.get_grid_dataset(states=np.array([100, 100]), actions=np.array([3]), step=True)
        self.visualize_gradient(dataset,
                                NeuralNetworkPolicy([50], [torch.relu], self.task, lambda x: 2 * torch.tanh(x)),
                                np.random.choice(np.arange(0, 201), 4, replace=False))

    def analyze_gradient_uniform_dataset(self, n=20):
        dataset = task.environment.get_grid_dataset(states=np.array([25, 25]), actions=np.array([2]), step=True)
        dataset_generator = lambda: dataset
        policy_generator = lambda: NeuralNetworkPolicy([50], [torch.relu], self.task, lambda x: 2 * torch.tanh(x))
        self.gradient_statistics(dataset_generator, policy_generator, n=n)

    def onpolicy_bias_variance_estimates(self, confidence=5):
        self.print("The dataset is generated with rollout starting from the bottom position (as prescribed in the task)"
                   "and following the evaluation policy")
        n_rollout = 1
        random_start_task = RLTask(Pendulum2D(), gamma=0.95, max_episode_length=200)

        for policy in self.policies:
            def get_dataset():
                dataset = self.task.get_empty_dataset(n_max_row=n_rollout * self.task.max_episode_length)
                collector = RLCollector(dataset, random_start_task, policy)
                collector.collect_rollouts(int(n_rollout))
                return dataset.train_ds
            analyzer = MCAnalyzer(self.task, policy)
            ground_truth = analyzer.get_return()
            self.bias_variance_return(get_dataset, policy, ground_truth, abs_confidence=confidence)



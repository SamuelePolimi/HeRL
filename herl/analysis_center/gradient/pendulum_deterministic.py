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
from herl.solver import RLCollector

task = RLTask(Pendulum2D(np.array([np.pi, 0.])), gamma=0.95, max_episode_length=200)


def _get_path(filename):
    full_path = os.path.realpath(__file__)
    path, _ = os.path.split(full_path)
    return path + "/" + filename


def _new_networks():
    return [NeuralNetworkPolicy([50], [torch.relu], task, lambda x: 2 * torch.tanh(x)) for i in range(3)]


def _generate_neural_networks():
    policies = _new_networks()
    [policy.save_model(_get_path("../../models/rnd_det_pendulum_%d.torch" %i)) for i, policy in enumerate(policies)]
    return policies


def _load_neural_networks():
    policies = _new_networks()
    [policy.load_model(_get_path("../../models/rnd_det_pendulum_%d.torch" % i)) for i, policy in enumerate(policies)]
    return policies


class Reset:

    def __init__(self):
        self.policies = None

    def reset(self):
        """
        By calling this method, you overwrite the current policy, and all the results will be computed and
        saved on disk.
        :return:
        """
        self.reset_policies()
        self.reset_value_function()

    def reset_policies(self):
        print("Policy reset.")
        self.policies = _generate_neural_networks()

    def reset_value_function(self):
        print("Value reset.")
        for i, policy in enumerate(self.policies):
            indexes = np.random.choice(range(len(self.policies[0].get_parameters())), 5, False)
            indexes = np.sort(indexes)
            np.save("pendulum_deterministic_policy_%d.npy" % i, indexes)
            analyzer = MCAnalyzer(task, policy)
            visualizer = ReturnRowVisualizer()
            visualizer.compute(analyzer, policy, indexes, [0.1] * 5, [200] * 5)
            visualizer.save(_get_path("../../plots/landscape_pendulum_%d.npz" % i))
#
# reset = Reset()
# reset.reset()

policies = _load_neural_networks()
policy = policies[0]
indexes = np.load("pendulum_deterministic_policy_0.npy")
visualizer = RowVisualizer.load(_get_path("../../plots/landscape_pendulum_0.npz"))
fig, axs = plt.subplots(1, 5)
visualizer.visualize(axs)
visualizer.visualize_decorations(axs)
mc = MCAnalyzer(task, policy)
gr_vis = GradientRowVisualizer()
gr_vis.compute(policy, mc, [0.02]*5, indexes)
gr_vis.visualize(axs)
plt.show()


class Pendulum2DGradientAnalyzer(GradientAnalyzer):

    def __init__(self, *algorithm_constructors):
        """

        :param algorithm_constructors:
        """
        GradientAnalyzer.__init__(self, task, analyzer, *algorithm_constructors)
        self.policies = policies
        self.print("""Pendulum2DGradientAnalyzer.
        The purpose of this analyzer is to evaluate the gradient estimation of a deterministic 2D pendulum.""")

    def all_analysis(self):
        pass

    def visualize_gradient_uniform_dataset(self, discretization=50, **graphic_args):

        dataset = task.environment.get_grid_dataset(states=np.array([25, 25]), actions=np.array([2]), step=True)
        self.visualize_gradient(dataset,
                                NeuralNetworkPolicy([50], [torch.relu], self.task, lambda x: 2 * torch.tanh(x)),
                                np.random.choice(np.arange(0, 201), 4, replace=False))

    def analyze_gradient_uniform_dataset(self, n=20):
        dataset = task.environment.get_grid_dataset(states=np.array([25, 25]), actions=np.array([2]), step=True)
        dataset_generator = lambda: dataset
        policy_generator = lambda: NeuralNetworkPolicy([50], [torch.relu], self.task, lambda x: 2 * torch.tanh(x))
        self.gradient_statistics(dataset_generator, policy_generator, n=n)


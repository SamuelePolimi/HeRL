"""
This analyser provides a neural network randomly initialized, and evaluate the prediction of the algorithm under
different tests, plotting the value function, the return under different datasets.
"""
import torch, numpy as np
import matplotlib.pyplot as plt
import os

from herl.actor import NeuralNetworkPolicy
from herl.dataset import Dataset, Domain, Variable
from herl.rl_interface import RLTask, Critic, Online, PolicyGradient, Offline
from herl.classic_envs import Pendulum2D
from herl.rl_analysis import MCAnalyzer
from herl.rl_visualizer import plot_value, plot_value_row, plot_state_cloud, plot_state_distribution
from herl.analysis_center.critic.critic_analyzer import CriticAnalyzer

task = RLTask(Pendulum2D(np.array([np.pi, 0.])), gamma=0.95, max_episode_length=200)


def _get_path(filename):
    full_path = os.path.realpath(__file__)
    path, _ = os.path.split(full_path)
    return path + "/" + filename


def _new_network():
    return NeuralNetworkPolicy([50], [torch.relu], task, lambda x: 2 * torch.tanh(x))


def _generate_neural_network():
    policy = _new_network()
    policy.save_model(_get_path("../../models/rnd_det_pendulum.torch"))
    return policy


def _load_neural_network():
    policy = _new_network()
    policy.load_model(_get_path("../../models/rnd_det_pendulum.torch"))
    return policy


class Reset:

    def __init__(self):
        self.policy = None

    def reset(self):
        """
        By calling this method, you overwrite the current policy, and all the results will be computed and
        saved on disk.
        :return:
        """
        self.reset_policy()
        self.reset_value_function()

    def reset_policy(self):
        print("Policy reset.")
        self.policy = _generate_neural_network()

    def reset_value_function(self):
        print("Value reset.")
        analyzer = MCAnalyzer(task, self.policy)
        dataset = task.environment.get_grid_dataset(np.array([100, 100]))
        states = dataset.get_full()["state"]
        results = analyzer.get_V(states)
        dataset = Dataset(Domain(Variable("state", 2), Variable("value", 1)), n_max_row=results.shape[0])
        dataset.notify_batch(state=states, value=results)
        dataset.save(_get_path("../../datasets/pendulum2d/rnd_det.npz"))


class SavedAnalyzer(Critic, Online, PolicyGradient):

    def __init__(self, task, policy):
        Critic.__init__(self, "MC")
        Online.__init__(self, "MC", task)
        PolicyGradient.__init__(self, "MC", policy)

    def get_V(self, states):
        dataset = Dataset.load(_get_path("../../datasets/pendulum2d/rnd_det.npz"),
                               Domain(Variable("state", 2), Variable("value", 1)))
        return dataset.get_full()["value"]


# Reset().reset()

policy = _load_neural_network()
analyzer = SavedAnalyzer(task, policy)


class Pendulum2DCriticAnalyzer(CriticAnalyzer):

    def __init__(self, *algorithm_constructors):
        CriticAnalyzer.__init__(self, task, analyzer, *algorithm_constructors)
        self.policy = policy

    def visualize_value_small_uniform_dataset(self, *discretization, **graphic_args):
        dataset = task.environment.get_grid_dataset(states=np.array([25, 25]), actions=np.array([2]), step=True)
        ax = plt.subplot()
        plot_state_distribution(ax, task.environment, dataset, discretization=np.array([100, 100]), bandwidth=0.3)
        plot_state_cloud(ax, dataset, s=1.)
        l = ax.set_ylabel(self.task.environment.state_space.symbol[1])
        l.set_rotation(0)
        ax.set_xlabel(self.task.environment.state_space.symbol[0])
        ax.set_title("Dataset")
        plt.show()

        if len(discretization)==0:
            self.visualize_value(dataset, self.policy, *([np.array([100, 100])]*len(self.algorithm_constructors)),
                                 **graphic_args)
        else:
            self.visualize_value(dataset, self.policy, *discretization, **graphic_args)



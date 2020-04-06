import torch, numpy as np
import matplotlib.pyplot as plt
from typing import Union

from herl.actor import NeuralNetworkPolicy
from herl.dataset import Dataset, Domain, Variable
from herl.rl_interface import RLTask, Critic, Online, PolicyGradient, Offline
from herl.classic_envs import Pendulum2D
from herl.rl_analysis import MCAnalyzer
from herl.rl_visualizer import plot_value, plot_value_row


class CriticAnalyzer:

    def __init__(self, task, reference, algorithm_constructors):
        """

        :param task:
        :type task: RLTask
        """
        self.task = task
        self.reference = reference
        self.tak_descriptor = task.get_descriptor()
        self.algorithm_constructors = algorithm_constructors

    def visualize_value(self, dataset, policy, *discretization, **graphic_args):
        algos = [constructor(task=self.tak_descriptor, dataset=dataset, policy=policy)
                 for constructor in self.algorithm_constructors]
        fig, ax = plt.subplots(1, 1 + len(algos))
        plot_value_row(fig, ax, self.task.environment,
                            [self.reference] + algos, [np.array([100, 100])] + list(discretization), **graphic_args)
        plt.show()

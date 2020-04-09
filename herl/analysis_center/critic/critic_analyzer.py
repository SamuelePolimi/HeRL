import torch, numpy as np
import matplotlib.pyplot as plt
from typing import Union

from herl.actor import NeuralNetworkPolicy
from herl.dataset import Dataset, Domain, Variable
from herl.rl_interface import RLTask, Critic, Online, PolicyGradient, Offline
from herl.classic_envs import Pendulum2D
from herl.rl_analysis import MCAnalyzer, BaseAnalyzer, bias_variance_estimate
from herl.rl_visualizer import plot_value, plot_value_row, plot_state_distribution, plot_state_cloud


class CriticAnalyzer(BaseAnalyzer):

    def __init__(self, task, reference, algorithm_constructors, verbose=True):
        """
        This class analyzes the most important quantities for a critic.
        :param task:
        :type task: RLTask
        """
        BaseAnalyzer.__init__(self, verbose, True)
        self.task = task
        self.reference = reference
        self.tak_descriptor = task.get_descriptor()
        self.algorithm_constructors = algorithm_constructors

    def visualize_value(self, dataset, policy, discretization_reference, *discretization, **graphic_args):
        self.print("Visualization of the dataset fed to the algorithm.")
        ax = plt.subplot()
        plot_state_distribution(ax, self.task.environment, dataset, discretization=np.array([100, 100]), bandwidth=0.3)
        plot_state_cloud(ax, dataset, s=1.)
        l = ax.set_ylabel(self.task.environment.state_space.symbol[1])
        l.set_rotation(0)
        ax.set_xlabel(self.task.environment.state_space.symbol[0])
        ax.set_title("Dataset")
        self.show()

        self.print("Visualization of the value estimation.")
        algos = [constructor(task=self.tak_descriptor, dataset=dataset, policy=policy)
                 for constructor in self.algorithm_constructors]

        fig, ax = plt.subplots(1, 1 + len(algos))
        if len(discretization)==0:
            discretization_list = [discretization_reference]*len(self.algorithm_constructors)
        else:
            discretization_list = list(discretization)
        plot_value_row(fig, ax, self.task.environment,
                            [self.reference] + algos, [discretization_reference] + discretization_list, **graphic_args)
        self.show()

    def bias_variance_return(self, dataset_generator, policy, abs_confidence=10.):
        self.print("We use different datasets generated with a random policy, and evaluate the bias and the variance of "
                   "the return's estimator.")
        ret = self.reference.get_return()
        names = [constructor(task=self.tak_descriptor, dataset=dataset_generator(), policy=policy).name
                 for constructor in self.algorithm_constructors]
        algos = [lambda: constructor(task=self.tak_descriptor, dataset=dataset_generator(), policy=policy).get_return()
                 for constructor in self.algorithm_constructors]
        estimates = [bias_variance_estimate(ret, alg, abs_confidence=abs_confidence) for alg in algos]
        fig, ax = plt.subplots(1, len(names))
        for (bias, variance, estimates, _), name in zip(estimates, names):
            ax.set_title("%s estimate" % name)
            ax.violinplot([estimates], showmeans=True)
            ax.set_ylabel("$\hat{J}_{%s}$" % name)
            ax.set_xlim(0, 2)
            ax.set_xticks([])
            ax.scatter(np.ones_like(estimates)*1.1, estimates, s=10, c="green", label="Estimates")
            ax.scatter(1.1, ret, s=15, c="orange", label="Ground truth")
            ax.legend(loc="best")
            self.print("Estimator %s has a bias of %f and variance of %f with confidence of 95%%."\
                       % (name, bias, variance))
        self.show()

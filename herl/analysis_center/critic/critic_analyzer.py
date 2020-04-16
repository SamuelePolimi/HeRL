import torch, numpy as np
import matplotlib.pyplot as plt
from typing import Union

from herl.actor import NeuralNetworkPolicy
from herl.dataset import Dataset, Domain, Variable
from herl.rl_interface import RLTask, Critic, Online, PolicyGradient, Offline
from herl.classic_envs import Pendulum2D
from herl.rl_analysis import MCAnalyzer, BaseAnalyzer, bias_variance_estimate
from herl.rl_visualizer import plot_value, plot_value_row, plot_state_distribution, plot_state_cloud, RowVisualizer, ValueFunctionVisualizer


class CriticAnalyzer(BaseAnalyzer):

    def __init__(self, task, algorithm_constructors, verbose=True):
        """
        This class analyzes the most important quantities for a critic.
        :param task:
        :type task: RLTask
        """
        BaseAnalyzer.__init__(self, verbose, True)
        self.task = task
        self.tak_descriptor = task.get_descriptor()
        self.algorithm_constructors = algorithm_constructors

    def visualize_value(self, value_visualizer, dataset, policy, *discretization, **graphic_args):

        self.print("Visualization of the value estimation.")
        algos = [constructor(task=self.tak_descriptor, dataset=dataset, policy=policy)
                 for constructor in self.algorithm_constructors]

        fig, axs = plt.subplots(1, 1 + len(algos))
        discretization_list = list(discretization)

        value_row_visualizer = RowVisualizer("value_row_visualizer")
        visualizers = []
        for algo, d in zip(algos, discretization_list):
            visualizer = ValueFunctionVisualizer()
            visualizer.compute(self.task.environment, algo, d)
            visualizers.append(visualizer)
        value_row_visualizer.sub_visualizer = [value_visualizer] + visualizers
        value_row_visualizer.visualize(axs)
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

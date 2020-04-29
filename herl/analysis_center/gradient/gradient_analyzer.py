import torch, numpy as np
import matplotlib.pyplot as plt
from typing import Union

from herl.solver import RLCollector
from sklearn.neighbors import KernelDensity

from herl.actor import NeuralNetworkPolicy
from herl.dataset import Dataset, Domain, Variable
from herl.rl_interface import RLTask, Critic, Online, PolicyGradient, Offline
from herl.classic_envs import Pendulum2D
from herl.rl_analysis import MCAnalyzer, BaseAnalyzer, bias_variance_estimate
from herl.rl_visualizer import GradientEstimateVisualizer, RowVisualizer, BiasVarianceVisualizer, \
    ParametricGradientEstimateVisualizer


class GradientAnalyzer(BaseAnalyzer):

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
        self._n_algorithms = len(self.algorithm_constructors)

    def visualize_off_policy_gradient_direction_estimates(self, policies, ground_truth, dataset):
        policy = policies[0]
        if self._n_algorithms == 1:
            fig, ax = plt.subplots(1, 1)
            axs = [ax]
        else:
            fig, axs = plt.subplots(1, self._n_algorithms)
        row = RowVisualizer("gradient_estimates_row")
        for constructor in self.algorithm_constructors:
            visualizer = GradientEstimateVisualizer()
            visualizer.unmute()
            #visualizer.compute(None, None, None)
            visualizer.compute(policies, ground_truth, constructor(self.task.get_descriptor(), dataset, policy))
            row.sub_visualizer.append(visualizer)
        row.visualize(axs)
        row.visualize_decorations(axs)
        self.show()
        return row

    def visualize_gradient_direction_samples(self, policies, ground_truth, get_dataset, samples_list):
        if self._n_algorithms == 1:
            fig, ax = plt.subplots(1, 1)
            axs = [ax]
        else:
            fig, axs = plt.subplots(1, self._n_algorithms)

        row = RowVisualizer("gradient_estimates_row")
        for constructor in self.algorithm_constructors:
            visualizer = ParametricGradientEstimateVisualizer()
            visualizer.unmute()
            visualizer.compute(policies,
                               ground_truth,
                               lambda x, y: constructor(self.task.get_descriptor(), get_dataset(y), x).get_gradient(),
                               samples_list)
            row.sub_visualizer.append(visualizer)
        row.visualize(axs)
        row.visualize_decorations(axs)
        self.show()
        return row

    def visualize_bias_variance_gradient(self, ground_truth, policy, dataset_generator, parameters,
                                                    confidence=10., min_samples=10, max_samples=1000, visualize=False):

        row = RowVisualizer("return_estimates")
        for algorithm in self.algorithm_constructors:
            visualizer = BiasVarianceVisualizer()
            estimator = lambda x: lambda: algorithm(self.task.get_descriptor(), dataset_generator(x),
                                                    policy).get_gradient()
            visualizer.compute(estimator, ground_truth, parameters, confidence=confidence, min_samples=min_samples,
                               max_samples=max_samples)
            row.sub_visualizer.append(visualizer)
        fig, axs = plt.subplots(1, len(self.algorithm_constructors))
        row.visualize([axs])
        if visualize:
            self.show()
        return row




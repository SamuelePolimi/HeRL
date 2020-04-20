import numpy as np
import matplotlib.pyplot as plt

from herl.rl_interface import RLTask
from herl.rl_analysis import BaseAnalyzer, bias_variance_estimate
from herl.rl_visualizer import RowVisualizer, ValueFunctionVisualizer, BiasVarianceVisualizer, EstimatesVisualizer, \
    SingleEstimatesVisualizer


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

    def visualize_return_estimates(self, ground_truth, policy, dataset_generator):
        row = RowVisualizer("return_estimates")
        for algorithm in self.algorithm_constructors:
            visualizer = SingleEstimatesVisualizer()
            estimator = lambda: algorithm(self.task.get_descriptor(), policy, dataset_generator())
            visualizer.compute(estimator, ground_truth)
            row.sub_visualizer.append(visualizer)
        fig, axs = plt.subplots(1, len(self.algorithm_constructors))
        row.visualize(axs)
        self.show()

    def visualize_parametrized_return_estimates(self, ground_truth, policy, dataset_generator, parameters):
        row = RowVisualizer("return_estimates")
        for algorithm in self.algorithm_constructors:
            visualizer = EstimatesVisualizer()
            estimator = lambda x: algorithm(self.task.get_descriptor(), policy, dataset_generator(x))
            visualizer.compute(estimator, ground_truth, parameters)
            row.sub_visualizer.append(visualizer)
        fig, axs = plt.subplots(1, len(self.algorithm_constructors))
        row.visualize(axs)
        self.show()

    def visualize_bias_variance(self, ground_truth, policy, dataset_generator, parameters):
        pass


    def bias_variance_return(self, dataset_generator, policy, ground_truth, abs_confidence=10.):
        self.print("We use different datasets generated with a random policy, and evaluate the bias and the variance of "
                   "the return's estimator.")
        names = [constructor(task=self.tak_descriptor, dataset=dataset_generator(), policy=policy).name
                 for constructor in self.algorithm_constructors]
        algos = [lambda: constructor(task=self.tak_descriptor, dataset=dataset_generator(), policy=policy).get_return()
                 for constructor in self.algorithm_constructors]
        estimates = [bias_variance_estimate(ground_truth, alg, abs_confidence=abs_confidence) for alg in algos]
        fig, ax = plt.subplots(1, len(names))
        for (bias, variance, estimates, _), name in zip(estimates, names):
            ax.set_title("%s estimate" % name)
            ax.violinplot([estimates], showmeans=True)
            ax.set_ylabel("$\hat{J}_{%s}$" % name)
            ax.set_xlim(0, 2)
            ax.set_xticks([])
            ax.scatter(np.ones_like(estimates)*1.1, estimates, s=10, c="green", label="Estimates")
            ax.scatter(1.1, ground_truth, s=15, c="orange", label="Ground truth")
            ax.legend(loc="best")
            self.print("Estimator %s has a bias of %f and variance of %f with confidence of 95%%."\
                       % (name, bias, variance))
        self.show()

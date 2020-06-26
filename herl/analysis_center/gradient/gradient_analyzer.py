import numpy as np

from herl.rl_interface import RLTask
from herl.rl_analysis import BaseAnalyzer
from herl.rl_visualizer import RowVisualizer, BiasVarianceVisualizer, \
    ParametricGradientEstimateVisualizer, Vector2DParametricDirection, Vector2DVisualizer
from herl.multiprocess import MultiProcess


class GradientAnalyzer(BaseAnalyzer):

    def __init__(self, task, verbose=True, **algorithm_constructors):
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

    def base_gradient_direction_samples(self, policies, ground_truth, get_dataset, samples_list, gradient_2d=False):

        row = RowVisualizer("gradient_estimates_row")
        for name in self.algorithm_constructors:
            constructor = self.algorithm_constructors[name]
            visualizer = Vector2DParametricDirection() if gradient_2d else ParametricGradientEstimateVisualizer()
            visualizer.unmute()
            visualizer.compute(policies,
                               ground_truth,
                               lambda x, y: constructor(self.task.get_descriptor(), get_dataset(x, y), x).get_gradient(),
                               samples_list,
                               x_label="n samples",
                               title=name)
            row.sub_visualizer.append(visualizer)

        return row

    def base_bias_variance_gradient(self, ground_truth, policy, dataset_generator, parameters,
                                                    confidence=10., inner_samples=1, min_samples=10, max_samples=1000, visualize=False):

        row = RowVisualizer("return_estimates")
        for name in self.algorithm_constructors.keys():
            algorithm = self.algorithm_constructors[name]
            visualizer = BiasVarianceVisualizer()
            estimator = lambda x: lambda: algorithm(self.task.get_descriptor(), dataset_generator(x),
                                                    policy).get_gradient()
            visualizer.compute(estimator, ground_truth, parameters, confidence=confidence, inner_samples=inner_samples,
                               min_samples=min_samples,
                               max_samples=max_samples)
            row.sub_visualizer.append(visualizer)

        return row


class OffPolicyGradientAnalyzer(BaseAnalyzer):

    def __init__(self, task, verbose=True, **algorithm_constructors):
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

    def base_off_policy_bias_variance_gradient(self, ground_truth, policy, dataset_generator, parameters,
                                                    confidence=10., inner_samples=1, min_samples=10, max_samples=1000,
                                               visualize=False):

        row = RowVisualizer("return_estimates")
        p = self.get_progress_bar("Bias/Variance Computation", max_iter=len(self.algorithm_constructors))
        for name in self.algorithm_constructors.keys():
            algorithm = self.algorithm_constructors[name]
            visualizer = BiasVarianceVisualizer()

            def parametric_estimates(x):
                dataset, behavior = dataset_generator(x)
                return algorithm(self.task.get_descriptor(), dataset,
                          policy, behavior).get_gradient()

            def estimator(x):
                def inner_estimator():
                    return parametric_estimates(x)
                return inner_estimator

            visualizer.compute(estimator, ground_truth, parameters, confidence=confidence, inner_samples=inner_samples,
                               min_samples=min_samples,
                               max_samples=max_samples)
            p.notify()
            row.sub_visualizer.append(visualizer)
        return row

    def base_test_gradient(self, policy, behavior, ground_truth, dataset_generator, n_samples=100):
        row = RowVisualizer("return_estimates")
        mp = MultiProcess()

        p = self.get_progress_bar("gradient_test", max_iter=len(self.algorithm_constructors))
        for name, algorithm in self.algorithm_constructors.items():

            def estimator():
                return algorithm(self.task.get_descriptor(), dataset_generator(),
                          policy, behavior).get_gradient()

            process = lambda: estimator()

            estimates = mp.compute(process, n_samples)

            p.notify()

            visualizer = Vector2DVisualizer()
            params = {'Estimates': np.array(estimates),
                      'Ground thruth': np.array([ground_truth])}
            visualizer.compute(**params)
            row.sub_visualizer.append(visualizer)
        return row

    def base_off_policy_gradient_direction_samples(self, policy, ground_truth, get_dataset, samples_list,
                                                   variable_name="n samples", gradient_2d=False, n_policies=100):

        row = RowVisualizer("gradient_estimates_row")

        def estimator(x, y):
            dataset, behavior = get_dataset(x, y)
            return constructor(self.task.get_descriptor(), dataset, x, behavior).get_gradient()

        p = self.get_progress_bar("Parametric Gradient Direction", max_iter=len(self.algorithm_constructors))
        for name in self.algorithm_constructors:
            constructor = self.algorithm_constructors[name]
            visualizer = Vector2DParametricDirection() if gradient_2d else ParametricGradientEstimateVisualizer()
            visualizer.compute(policy,
                               ground_truth,
                               estimator,
                               samples_list,
                               x_label=variable_name,
                               title=name,
                               n_estimates=n_policies)
            p.notify()
            row.sub_visualizer.append(visualizer)

        return row


class ParametricOffPolicyGradientAnalyzer(BaseAnalyzer):

    def __init__(self, task, verbose=True, **algorithm_constructors):
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

    def base_off_policy_bias_variance_gradient(self, ground_truth, policy, dataset_generator, parameters,
                                                    confidence=10., inner_samples=1, min_samples=10, max_samples=1000):

        row = RowVisualizer("return_estimates")
        for name in self.algorithm_constructors.keys():
            algorithm = self.algorithm_constructors[name]
            visualizer = BiasVarianceVisualizer()
            def parametric_estimates(x):
                dataset, behavior = dataset_generator()
                return algorithm(self.task.get_descriptor(), dataset,
                          policy, behavior, x).get_gradient()

            estimator = lambda x: lambda: parametric_estimates(x)
            visualizer.compute(estimator, ground_truth, parameters, confidence=confidence, inner_samples=inner_samples,
                               min_samples=min_samples,
                               max_samples=max_samples)
            row.sub_visualizer.append(visualizer)
        return row

    def base_test_gradient(self, policy, behavior, ground_truth, dataset_generator, n_samples=100, parameter=None):
        row = RowVisualizer("return_estimates")
        for name, algorithm in self.algorithm_constructors.items():
            p = self.get_progress_bar("gradient_test %s" % name, max_iter=n_samples)
            estimates = []
            def estimator():
                return algorithm(self.task.get_descriptor(), dataset_generator(),
                          policy, behavior, parameter).get_gradient()
            for _ in range(n_samples):
                estimates.append(estimator())
                p.notify()
            visualizer = Vector2DVisualizer()
            visualizer.compute(estimates=np.array(estimates), ground_truth=np.array([ground_truth]))
            row.sub_visualizer.append(visualizer)
        return row

    def base_off_policy_gradient_direction_samples(self, policies, ground_truth, get_dataset, samples_list,
                                                   variable_name="n_samples", gradient_2d=False):

        row = RowVisualizer("gradient_estimates_row")
        # TODO: check here
        def estimator(x, y):
            dataset, behavior = get_dataset(x)
            return constructor(self.task.get_descriptor(), dataset, x, behavior).get_gradient()
        for name in self.algorithm_constructors:
            constructor = self.algorithm_constructors[name]
            visualizer = Vector2DParametricDirection() if gradient_2d else ParametricGradientEstimateVisualizer()
            visualizer.unmute()
            visualizer.compute(policies,
                               ground_truth,
                               estimator,
                               samples_list,
                               x_label=variable_name,
                               title=name)
            row.sub_visualizer.append(visualizer)

        return row
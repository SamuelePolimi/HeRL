"""
This analyser provides a neural network randomly initialized, and evaluate the prediction of the algorithm under
different tests, plotting the value function, the return under different datasets.
"""
import torch, numpy as np
import matplotlib.pyplot as plt
import os

from herl.actor import NeuralNetworkPolicy, UniformPolicy
from herl.rl_interface import RLTask, DeterministicState
from herl.classic_envs import Pendulum2D
from herl.rl_analysis import MCAnalyzer
from herl.rl_visualizer import ValueFunctionVisualizer, PlotVisualizer
from herl.analysis_center.critic.critic_analyzer import CriticAnalyzer
from herl.solver import RLCollector
from herl.rl_analysis import Printable

task = RLTask(Pendulum2D(), DeterministicState(np.array([np.pi, 0.])), gamma=0.95, max_episode_length=200)


def _get_path(filename):
    full_path = os.path.realpath(__file__)
    path, _ = os.path.split(full_path)
    return path + "/" + filename


def _new_network():
    return NeuralNetworkPolicy([50], [torch.relu], task, lambda x: 2 * torch.tanh(x))


def _generate_neural_networks():
    policies = [_new_network() for _ in range(10)]
    for i, policy in enumerate(policies):
        policy.save(_get_path("pendulum_deterministic_data/agents/rnd_det_pendulum_%d.torch" % i))
    return policies


def _load_neural_networks():
    policies = [_new_network() for _ in range(10)]
    for i, policy in enumerate(policies):
        policy.load(_get_path("pendulum_deterministic_data/agents/rnd_det_pendulum_%d.torch" % i))
        policy.symbol = "\pi_{%d}" % i
    return policies


class Reset(Printable):

    def __init__(self):
        Printable.__init__(self, "Reset")
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
        self.policies = _generate_neural_networks()

    def reset_value_function(self):
        print("Value reset.")
        bar = self.get_progress_bar("Compute values", 10)
        for i, policy in enumerate(self.policies):
            bar.notify()
            analyzer = MCAnalyzer(task, policy)
            visualizer = ValueFunctionVisualizer()
            visualizer.compute(task.environment, analyzer, np.array([200, 200]))
            visualizer.save(_get_path("pendulum_deterministic_data/plots/pendulum_deterministic_value_%d.npz" % i))


policies = _load_neural_networks()


class Pendulum2DCriticAnalyzer(CriticAnalyzer):

    def __init__(self, *algorithm_constructors):
        """
        This class analyzes the performance of the critic on a simple deterministic 2D Pendulum.
        The task consists in evaluating a neural network policy given a set of data.
        The analyzer includes:
             - small and large uniform data to evaluate the policy
             - dataset generated randomply to evaluate the policy
             - bias and variance of the estimated return
             - return landscape
        :param algorithm_constructors:
        """
        CriticAnalyzer.__init__(self, task, algorithm_constructors)
        self.policies = policies
        self.print("""Pendulum2DCriticAnalyzer.
        The purpose of this analyzer is to evaluate the critic estimation of a deterministic 2D pendulum.""")

    def all_analysis(self):
        self.visualize_value_small_uniform_dataset()
        self.visualize_value_large_uniform_dataset()
        self.visualize_value_random_policy_dataset()
        self.onpolicy_bias_variance_estimates()
        self.offpolicy_bias_variance_estimates()

    def visualize_value_small_uniform_dataset(self, *discretization, **graphic_args):
        self.print("The dataset is generated on a grid of values 25x25x2 (angle, velocity, action).")
        dataset = task.environment.get_grid_dataset(states=np.array([25, 25]), actions=np.array([2]), step=True)
        for i, policy in enumerate(self.policies):
            self.visualize_value(PlotVisualizer.load(
                _get_path("pendulum_deterministic_data/plots/pendulum_deterministic_value_%d.npz" % i)),
                                 dataset, policy, *discretization, **graphic_args)

    def visualize_value_large_uniform_dataset(self, *discretization, **graphic_args):
        self.print("The dataset is generated on a grid of values 50x50x3 (angle, velocity, action).")
        dataset = task.environment.get_grid_dataset(states=np.array([50, 50]), actions=np.array([3]), step=True)

        for i, policy in enumerate(self.policies):
            self.visualize_value(PlotVisualizer.load(
                _get_path("pendulum_deterministic_data/plots/pendulum_deterministic_value_%d.npz" % i)),
                dataset, policy, *discretization, **graphic_args)

    def visualize_value_random_policy_dataset(self, *discretization, **graphic_args):
        self.print("The dataset is generated with rollout starting half o "
                   "position and following a random policy")
        uniform_policy = UniformPolicy(np.array([-2.]), np.array([2.]))
        n_rollout = 1
        random_start_task = RLTask(Pendulum2D(), gamma=0.95, max_episode_length=200)
        dataset = self.task.get_empty_dataset(n_max_row=n_rollout*self.task.max_episode_length)
        collector = RLCollector(dataset, random_start_task, uniform_policy)
        collector.collect_rollouts(int(n_rollout))

        for i, policy in enumerate(self.policies):
            self.visualize_value(PlotVisualizer.load(
                _get_path("pendulum_deterministic_data/plots/pendulum_deterministic_value_%d.npz" % i)),
                dataset.train_ds, policy, *discretization, **graphic_args)

    def offpolicy_bias_variance_estimates(self, confidence=5.):
        self.print("The dataset is generated with rollout starting from random position"
                   "and following a random policy")
        uniform_policy = UniformPolicy(np.array([-2.]), np.array([2.]))
        n_rollout = 20
        random_start_task = RLTask(Pendulum2D(), gamma=0.95, max_episode_length=200)

        for policy in self.policies:
            def get_dataset():
                dataset = self.task.get_empty_dataset(n_max_row=n_rollout * self.task.max_episode_length)
                collector = RLCollector(dataset, random_start_task, uniform_policy)
                collector.collect_rollouts(int(n_rollout))
                return dataset.train_ds
            analyzer = MCAnalyzer(self.task, policy)
            ground_truth = analyzer.get_return()
            self.bias_variance_return(get_dataset, policy, ground_truth, abs_confidence=confidence)

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

    def onpolicy_sample_bias_variance_estimates(self):
        self.print("The dataset is generated with rollout starting from the bottom position (as prescribed in the task)"
                   " and following the evaluation policy")
        for policy in self.policies:
            analyzer = MCAnalyzer(self.task, policy)
            ground_truth = analyzer.get_return()

            def get_dataset(n):
                random_start_task = RLTask(Pendulum2D(), gamma=0.95, max_episode_length=200)
                dataset = self.task.get_empty_dataset(n_max_row=int(n))
                collector = RLCollector(dataset, random_start_task, policy)
                collector.collect_samples(int(n))
                return dataset.train_ds

            self.visualize_parametrized_return_estimates(ground_truth, policy, lambda x: get_dataset(x),
                                                         [250, 500, 1000, 1500, 2000, 3000, 5000], confidence=5.)
            self.show()

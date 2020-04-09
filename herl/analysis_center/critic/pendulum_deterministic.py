"""
This analyser provides a neural network randomly initialized, and evaluate the prediction of the algorithm under
different tests, plotting the value function, the return under different datasets.
"""
import torch, numpy as np
import matplotlib.pyplot as plt
import os

from herl.actor import NeuralNetworkPolicy, UniformPolicy
from herl.dataset import Dataset, Domain, Variable
from herl.rl_interface import RLTask, Critic, Online, PolicyGradient, Offline
from herl.classic_envs import Pendulum2D
from herl.rl_analysis import MCAnalyzer
from herl.rl_visualizer import sample_vs_bias_variance
from herl.analysis_center.critic.critic_analyzer import CriticAnalyzer
from herl.solver import RLCollector

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

    def get_return(self):
        analyzer = MCAnalyzer(task, self.policy)
        return analyzer.get_return()


policy = _load_neural_network()
analyzer = SavedAnalyzer(task, policy)


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
        CriticAnalyzer.__init__(self, task, analyzer, *algorithm_constructors)
        self.policy = policy
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

        self.visualize_value(dataset, self.policy, np.array([100, 100]), *discretization, **graphic_args)

    def visualize_value_large_uniform_dataset(self, *discretization, **graphic_args):
        self.print("The dataset is generated on a grid of values 50x50x3 (angle, velocity, action).")
        dataset = task.environment.get_grid_dataset(states=np.array([50, 50]), actions=np.array([3]), step=True)

        self.visualize_value(dataset, self.policy, np.array([100, 100]), *discretization, **graphic_args)

    def visualize_value_random_policy_dataset(self, *discretization, **graphic_args):
        self.print("The dataset is generated with rollout starting half o "
                   "position and following a random policy")
        uniform_policy = UniformPolicy(np.array([-2.]), np.array([2.]))
        n_rollout = 20
        random_start_task = RLTask(Pendulum2D(), gamma=0.95, max_episode_length=200)
        dataset = self.task.get_empty_dataset(n_max_row=n_rollout*self.task.max_episode_length)
        collector = RLCollector(dataset, random_start_task, uniform_policy)
        collector.collect_rollouts(int(n_rollout))

        self.visualize_value(dataset.train_ds, self.policy, np.array([100, 100]), *discretization, **graphic_args)

    def offpolicy_bias_variance_estimates(self):
        self.print("The dataset is generated with rollout starting from random position"
                   "and following a random policy")
        uniform_policy = UniformPolicy(np.array([-2.]), np.array([2.]))
        n_rollout = 20
        random_start_task = RLTask(Pendulum2D(), gamma=0.95, max_episode_length=200)

        def get_dataset():
            dataset = self.task.get_empty_dataset(n_max_row=n_rollout * self.task.max_episode_length)
            collector = RLCollector(dataset, random_start_task, uniform_policy)
            collector.collect_rollouts(int(n_rollout))
            return dataset.train_ds

        self.bias_variance_return(get_dataset, self.policy)

    def onpolicy_bias_variance_estimates(self):
        self.print("The dataset is generated with rollout starting from the bottom position (as prescribed in the task)"
                   "and following the evaluation policy")
        n_rollout = 1
        random_start_task = RLTask(Pendulum2D(), gamma=0.95, max_episode_length=200)

        def get_dataset():
            dataset = self.task.get_empty_dataset(n_max_row=n_rollout * self.task.max_episode_length)
            collector = RLCollector(dataset, random_start_task, self.policy)
            collector.collect_rollouts(int(n_rollout))
            return dataset.train_ds

        self.bias_variance_return(get_dataset, self.policy)

    def onpolicy_sample_bias_variance_estimates(self):
        self.print("The dataset is generated with rollout starting from the bottom position (as prescribed in the task)"
                   " and following the evaluation policy")
        n_rollout = 1
        ret = self.reference.get_return()

        def get_return_estimate(i, n):
            random_start_task = RLTask(Pendulum2D(), gamma=0.95, max_episode_length=200)
            dataset = self.task.get_empty_dataset(n_max_row=int(n))
            collector = RLCollector(dataset, random_start_task, self.policy)
            collector.collect_samples(int(n))
            return self.algorithm_constructors[i](self.task.get_descriptor(), dataset.train_ds, self.policy).get_return()

        ax = plt.subplot()
        sample_vs_bias_variance(ax, ret, lambda x: lambda: get_return_estimate(0, x), values=[1000, 1250, 1500, 2000, 5000])
        self.show()
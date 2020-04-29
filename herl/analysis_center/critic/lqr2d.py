"""
This analyser provides a neural network randomly initialized, and evaluate the prediction of the algorithm under
different tests, plotting the value function, the return under different datasets.
"""
import torch, numpy as np
import matplotlib.pyplot as plt
import os

from herl.actor import UniformPolicy, LinearPolicy
from herl.rl_interface import RLTask, DeterministicState
from herl.classic_envs import LQR
from herl.rl_analysis import MCAnalyzer
from herl.rl_visualizer import ValueFunctionVisualizer, PlotVisualizer, ReturnLandscape, StateCloudVisualizer\
    , StateDensityVisualizer
from herl.analysis_center.critic.critic_analyzer import CriticAnalyzer
from herl.solver import RLCollector
from herl.rl_analysis import Printable


def MyLQR():
    return LQR(A=np.array([[1.2, 0.],
                      [0, 1.1]]),
          B=np.array([[1., 0.],
                      [0., 1.]]),
          R=-np.array([[1., 0.],
                      [0., 1.]]),
          Q=-np.array([[0.4, 0.],
                      [0., 0.8]]),
          initial_state=np.array([-1., -1.]),
          state_box=np.array([2., 2.]),
          action_box=np.array([2., 2.]))

env = MyLQR()
task = RLTask(env, DeterministicState(np.array([-1., -1.])), gamma=0.5, max_episode_length=20)
# policy = LinearPolicy(2, 2, diagonal=True)
# policy.set_parameters(np.array([-1.25, -1.]))
# critic = MCAnalyzer(task, policy)
# visualizer = ValueFunctionVisualizer()
# visualizer.compute(env, critic, discretization=np.array([100, 100]))
# fig, ax = plt.subplots(1, 1)
# im = visualizer.visualize(ax)
# fig.colorbar(im, ax=ax)
# visualizer.visualize_x_label(ax)
# visualizer.visualize_title(ax)
# visualizer.visualize_y_label(ax)
# #fig.colorbar(ax)
# plt.show()
# visualizer = ReturnLandscape()
# fig, ax = plt.subplots(1, 1)
# visualizer.compute(critic, policy, [0, 1],
#                    np.array([-2.1, -2.1]),
#                    np.array([0.2, 0.2]),
#                    discretization=np.array([100, 100]))
# im = visualizer.visualize(ax)[0]
# fig.colorbar(im, ax=ax)
# visualizer.visualize_decorations(ax)
# plt.show()


def _sample_policy():
    policy = LinearPolicy(2, 2, diagonal=True)
    params = np.random.normal(loc=np.array([-1.2, -1.1]), scale=np.array([0.3, 0.3]))
    policy.set_parameters(params)
    return policy


def _sample_dataset(n_episodes=500):
    length = 20
    dataset = task.get_empty_dataset(n_max_row=length*n_episodes)
    for _ in range(n_episodes):
        policy = _sample_policy()
        collector = RLCollector(dataset, task, policy)
        collector.collect_rollouts(1)
    return dataset.train_ds

# fig, ax = plt.subplots(1, 1)
# ds = _sample_dataset()
# visualizer = StateDensityVisualizer()
# visualizer.compute(env, ds, 0.5, np.array([200, 200]))
# visualizer.visualize(ax)
# #visualizer.visualize_decorations(ax)
# visualizer = StateCloudVisualizer()
# visualizer.compute(ds)
# visualizer.visualize(ax, s=1)
# #visualizer.visualize_decorations(ax)
# plt.show()


def _get_path(filename):
    full_path = os.path.realpath(__file__)
    path, _ = os.path.split(full_path)
    return path + "/" + filename


def _generate_policies():
    policies = [_sample_policy() for _ in range(10)]
    for i, policy in enumerate(policies):
        policy.save(_get_path("lqr2d_data/agents/rnd_det_lqr_%d.torch" % i))
    return policies


def _load_policies():
    policies = [_sample_policy() for _ in range(10)]
    for i, policy in enumerate(policies):
        policy.load(_get_path("lqr2d_data/agents/rnd_det_lqr_%d.torch" % i))
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
        self.print("Policy reset.")
        self.policies = _generate_policies()

    def reset_value_function(self):
        self.print("Value reset.")
        bar = self.get_progress_bar("Compute values", 10)
        for i, policy in enumerate(self.policies):
            bar.notify()
            analyzer = MCAnalyzer(task, policy)
            visualizer = ValueFunctionVisualizer()
            visualizer.compute(task.environment, analyzer, np.array([200, 200]))
            visualizer.save(_get_path("lqr2d_data/plots/lqr_deterministic_value_%d.npz" % i))


# reset = Reset()
# reset.reset()
policies = _load_policies()


class LQR2DCriticAnalyzer(CriticAnalyzer):

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
        pass

    def offpolicy_visualize_value(self, *discretization, **graphic_args):
        self.print("The dataset is generated with rollout starting half o "
                   "position and following a random policy")

        dataset = _sample_dataset(100)

        for i, policy in enumerate(self.policies):
            self.visualize_value(PlotVisualizer.load(
                _get_path("lqr2d_data/plots/lqr_deterministic_value_%d.npz" % i)),
                dataset, policy, *discretization, **graphic_args)

    def offpolicy_bias_variance_estimates(self, confidence=5.):
        self.print("The dataset is generated with rollout starting from random position"
                   "and following a random policy")

        for policy in self.policies:
            analyzer = MCAnalyzer(self.task, policy)
            ground_truth = analyzer.get_return()
            self.bias_variance_return(lambda: _sample_dataset(100), policy, ground_truth, max_samples=10,
                                      abs_confidence=confidence)

    def offpolicy_sample_bias_variance_estimates(self):
        # TODO randomize initial state
        self.print("The dataset is generated with rollout starting from the bottom position (as prescribed in the task)"
                   " and following the evaluation policy")
        for policy in self.policies:
            analyzer = MCAnalyzer(self.task, policy)
            ground_truth = analyzer.get_return()

            self.visualize_parametrized_return_estimates(ground_truth, policy, lambda x: _sample_dataset(x),
                                                         [25, 50, 100, 200, 500, 750], confidence=5.)
            self.show()

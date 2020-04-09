import torch, numpy as np
import matplotlib.pyplot as plt
from typing import Union
from sklearn.neighbors import KernelDensity

from herl.actor import NeuralNetworkPolicy
from herl.dataset import Dataset, Domain, Variable
from herl.rl_interface import RLTask, Critic, Online, PolicyGradient, Offline
from herl.classic_envs import Pendulum2D
from herl.rl_analysis import MCAnalyzer, BaseAnalyzer, bias_variance_estimate
from herl.rl_visualizer import plot_value, plot_value_row, plot_state_distribution, plot_state_cloud, plot_gradient_row, plot_return_row


class GradientAnalyzer(BaseAnalyzer):

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

    def visualize_gradient(self, dataset, policy, param_indexes, discretization_reference=50, radius=0.5,
                           discretization=50, **graphic_args):
        self.print("Visualization of the gradient estimation for a random initialization of the neural network"
                   "and random inspection of the gradient.")
        backup_policy = self.reference.policy
        self.reference.policy = policy
        fig, axs = plt.subplots(1 + len(self.algorithm_constructors), len(param_indexes))
        ret = self.reference.get_return()
        gradient = self.reference.get_gradient()
        plot_return_row(axs[0, :], self.reference, param_indexes, radius, discretization_reference, **graphic_args)
        plot_gradient_row(axs[0, :], self.reference, param_indexes, ret, gradient=gradient)
        for ax, alg_con in zip(axs[1:, :], self.algorithm_constructors):
            algorithm = alg_con(self.task.get_descriptor(), dataset, policy)
            plot_return_row(ax, self.reference, param_indexes, radius, discretization_reference, **graphic_args)
            gradient = algorithm.get_gradient()
            plot_gradient_row(ax, algorithm, param_indexes, ret, gradient=gradient)
        self.reference.policy = backup_policy
        self.show()

    def gradient_statistics(self,  dataset_generator, policy_generator, n=20):
        backup_policy = self.reference.policy
        ground_truth = []
        estimations = {}
        scalar_products = {}
        normalized_scalar_products = {}

        for i in range(n):
            self.reference.policy = policy_generator()
            gradient = self.reference.get_gradient()
            ground_truth.append(gradient)
            for algorithm_generator in self.algorithm_constructors:
                algorithm = algorithm_generator(self.task.get_descriptor(), dataset_generator(), self.reference.policy)
                gradient = algorithm.get_gradient()
                if algorithm.name in estimations.keys():
                    estimations[algorithm.name].append(gradient)
                    scalar_products[algorithm.name].append(np.dot(ground_truth[i], gradient))
                    normalized_scalar_products[algorithm.name].append(np.dot(ground_truth[i]/np.linalg.norm(ground_truth[i]),
                                                                             gradient/np.linalg.norm(gradient)))
                else:
                    estimations[algorithm.name] = [gradient]
                    scalar_products[algorithm.name] = [np.dot(ground_truth[i], gradient)]
                    normalized_scalar_products[algorithm.name] = [np.dot(ground_truth[i]/np.linalg.norm(ground_truth[i]),
                                                                             gradient/np.linalg.norm(gradient))]

        self.print("The mean magnitude of the ground-truth gradients is %f" % np.mean(np.square(ground_truth)))
        mean_scalar_product = {}
        positives = {}
        fig, axs = plt.subplots(1, len(self.algorithm_constructors))
        axs = np.reshape(axs, len(self.algorithm_constructors))
        for alg_name, ax in zip(estimations.keys(), axs):
            mean_scalar_product[alg_name] = np.asscalar(np.mean(scalar_products[alg_name]))
            positives[alg_name] = np.asscalar(np.mean(np.array(scalar_products[alg_name]) >= 0))
            self.print("Algorithm %s has a mean scalar product of %f, "
                       "and the portion of positive products (gradients following the right direction) is of %f" %\
                       (alg_name, mean_scalar_product[alg_name], positives[alg_name]))
            self.print("The mean magnitude of the gradients estimation of %s is %f" %\
                       (alg_name, np.asscalar(np.mean(np.square(estimations[alg_name])))))
            ax.set_title("Mean normalized dot product of %s" % alg_name)
            ax.scatter(normalized_scalar_products[alg_name], -np.ones_like(normalized_scalar_products[alg_name]))
            kde = KernelDensity(bandwidth=0.1)
            kde.fit(np.array([normalized_scalar_products[alg_name]]).T)
            x_lin = np.linspace(-1, 1, 100)
            p = np.exp(kde.score_samples(x_lin.reshape(-1, 1)))
            ax.plot(x_lin, p.ravel())
            ax.axvline(0.)
            scalar_mean = np.asscalar(np.mean(normalized_scalar_products[alg_name]))
            ax.axvline(scalar_mean, color="green", label="Mean value")
            self.print("The mean value is %f, which indicates an avarege %s estimation of the gradient."\
                       % (scalar_mean, "good" if scalar_mean >= 0. else "bad"))
            ax.legend(loc="best")
        self.show()


        self.reference.policy = backup_policy




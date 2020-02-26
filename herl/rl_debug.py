import matplotlib.pyplot as plt
import numpy as np
import torch

from herl.rl_analysis import Analyser
from herl.dataset import Dataset, Domain, Variable
from herl.rl_interface import Critic, RLTask
from herl.utils import ConstantPolicyPendulum, Pendulum2D, RandomPolicyPendulum
from herl.datasets.library import search
from core.actor import Actor

class PolicyEvaluationPendulum2D(Analyser):

    def __init__(self, critic_class, dataset=None, policy=None, verbose=True, plot=True):

        Analyser.__init__(self, verbose, plot)
        self.critic_class = critic_class

        self.rl_task = RLTask(Pendulum2D(), gamma=0.99, max_episode_length=200)
        self.policy = Actor([50], [torch.relu], self.rl_task, lambda x: 2.0 * torch.tanh(x))
        self.dataset = dataset
        self.rl_algorithm = self.critic_class(self.rl_task, self.policy)  # type: Critic

    def analyze(self):
        self.print("Analysis Started")
        results = {}

        data = self.dataset.get_full()
        states = data["state"]
        values = data["value"]


        self.print("Value Prediction")
        predicted_values = self.rl_algorithm.get_V(state=states)


        self.print("Computation of the Mean Squared Error")
        results["mse"] = np.sqrt(np.mean(np.square(values.ravel() - predicted_values.ravel())).item())

        if self.plot:
            X = states[:, 0].reshape(100, 100)
            Y = states[:, 1].reshape(100, 100)
            V = values.reshape(100, 100)
            V_p = predicted_values.reshape(100, 100)
            plt.subplot(1, 2, 1)
            plt.title("True value function")
            plt.pcolormesh(X, Y, V)
            plt.colorbar()
            plt.subplot(1, 2, 2)
            plt.title("Predicted value function")
            plt.pcolormesh(X, Y, V_p)
            plt.colorbar()
            plt.show()

            
        self.print("The root mean squared error is %f" % results["mse"])

        return results


class PolicyEvaluationPendulum2DGridConstant(PolicyEvaluationPendulum2D):

    def __init__(self, critic_class, verbose=True, plot=True):

        dataset = search(Domain(Variable("state", 2), Variable("value", 1)),
                              "pendulum2d", "grid", "constant", "0", "0.95", "value")

        PolicyEvaluationPendulum2D.__init__(self, critic_class, dataset=dataset, policy=ConstantPolicyPendulum(),
                                            verbose=verbose, plot=plot)


class PolicyEvaluationPendulum2DConstant(PolicyEvaluationPendulum2D):

    def __init__(self, critic_class, verbose=True, plot=True):
        dataset = search(Domain(Variable("state", 2), Variable("value", 1)),
                         "pendulum2d", "sample_policy", "constant", "0", "0.95", "value")

        PolicyEvaluationPendulum2D.__init__(self, critic_class, dataset=dataset, policy=ConstantPolicyPendulum(),
                                            verbose=verbose, plot=False)


class PolicyEvaluationPendulum2DUniform(PolicyEvaluationPendulum2D):

    def __init__(self, critic_class, verbose=True, plot=True):
        dataset = search(Domain(Variable("state", 2), Variable("value", 1)),
                         "pendulum2d", "sample_policy", "uniform", "0.95", "value")

        PolicyEvaluationPendulum2D.__init__(self, critic_class, dataset=dataset, policy=RandomPolicyPendulum(),
                                            verbose=verbose, plot=False)


class PolicyEvaluationPendulum2DGridUniform(PolicyEvaluationPendulum2D):

    def __init__(self, critic_class, verbose=True, plot=True):
        dataset = search(Domain(Variable("state", 2), Variable("value", 1)),
                         "pendulum2d", "grid", "uniform", "0.95", "value")

        PolicyEvaluationPendulum2D.__init__(self, critic_class, dataset=dataset, policy=RandomPolicyPendulum(),
                                            verbose=verbose, plot=plot)
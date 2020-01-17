import matplotlib.pyplot as plt
import numpy as np

from herl.rl_analysis import Analyser
from herl.dataset import Dataset
from herl.rl_interface import Critic, RLTask
from herl.utils import ConstantPolicyPendulum, Pendulum2D
from herl.datasets.library import search


class PolicyEvaluationPendulum2D(Analyser):

    def __init__(self, critic_class, verbose=True, plot=True):

        Analyser.__init__(self, verbose, plot)
        self.critic_class = critic_class

        self.rl_task = RLTask(Pendulum2D(), gamma=0.95, max_episode_length=200)
        self.policy = ConstantPolicyPendulum()
        self.print("Loading the dataset...")
        self.dataset = search(self.rl_task, ["pendulum2d", "uniform", "constant", "0", "0.95", "value"])
        #Dataset.load("datasets/pendulum2d/constant_policy_0_uniform_state_v.npz", self.rl_task.domain)
        self.print("Dataset loaded...")
        self.rl_algorithm = self.critic_class(self.rl_task, self.policy)  # type: Critic

    def analyze(self):
        results = {}

        data = self.dataset.get_full()
        states = data["state"]
        values = data["value"]

        predicted_values = self.rl_algorithm.get_V(state=states)

        results["mse"] = np.mean(np.square(values.ravel() - predicted_values.ravel())).item()

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
            plt.show()

            plt.pcolormesh(X, Y, V)
        self.print("The mean squared error is %f" % results["mse"])

        return results



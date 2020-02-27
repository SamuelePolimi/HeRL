import matplotlib.pyplot as plt
import numpy as np
import torch.nn

from herl.rl_analysis import Analyser
from herl.dataset import Dataset, Domain, Variable
from herl.rl_interface import Critic, RLTask
from herl.utils import ConstantPolicyPendulum, Pendulum2D, RandomPolicyPendulum, MC2DPendulum, \
    RLUniformCollector2DPendulum
from herl.datasets.library import search
from herl.actor import Actor


class PolicyEvaluation(Analyser):

    def __init__(self, critic_class, rl_task, dataset=None, policy=None,
                 verbose=True, plot=True):

        Analyser.__init__(self, verbose, plot)
        self.critic_class = critic_class

        self.rl_task = rl_task
        self.policy = policy
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


class PolicyEvaluationPendulum2DGridConstant(PolicyEvaluation):

    def __init__(self, critic_class, verbose=True, plot=True):

        dataset = search(Domain(Variable("state", 2), Variable("value", 1)),
                              "pendulum2d", "grid", "constant", "0", "0.95", "value")

        PolicyEvaluation.__init__(self, critic_class, RLTask(Pendulum2D()), dataset=dataset, policy=ConstantPolicyPendulum(),
                                            verbose=verbose, plot=plot)


class PolicyEvaluationPendulum2DConstant(PolicyEvaluation):

    def __init__(self, critic_class, verbose=True, plot=True):
        dataset = search(Domain(Variable("state", 2), Variable("value", 1)),
                         "pendulum2d", "sample_policy", "constant", "0", "0.95", "value")

        PolicyEvaluation.__init__(self, critic_class, RLTask(Pendulum2D(), 0.95, 200), dataset=dataset, policy=ConstantPolicyPendulum(),
                                            verbose=verbose, plot=False)


class PolicyEvaluationPendulum2DUniform(PolicyEvaluation):

    def __init__(self, critic_class, verbose=True, plot=True):
        dataset = search(Domain(Variable("state", 2), Variable("value", 1)),
                         "pendulum2d", "sample_policy", "uniform", "0.95", "value")

        PolicyEvaluation.__init__(self, critic_class, RLTask(Pendulum2D(), 0.95, 200), dataset=dataset, policy=RandomPolicyPendulum(),
                                            verbose=verbose, plot=False)


class PolicyEvaluationPendulum2DGridUniform(PolicyEvaluation):

    def __init__(self, critic_class, verbose=True, plot=True):
        dataset = search(Domain(Variable("state", 2), Variable("value", 1)),
                         "pendulum2d", "grid", "uniform", "0.95", "value")

        PolicyEvaluation.__init__(self, critic_class, RLTask(Pendulum2D(), 0.95, 200), dataset=dataset, policy=RandomPolicyPendulum(),
                                            verbose=verbose, plot=plot)


class NeuralNetworkPolicyEvaluationPendulum2DGridUniform(PolicyEvaluation):

    def __init__(self, critic_class, verbose=True, plot=True):

        pendulum = Pendulum2D(initial_state=np.array([-np.pi, 0.]))
        rl_task = RLTask(pendulum, 0.95, 200)
        policy = Actor([50], [torch.nn.functional.relu], rl_task=rl_task)
        ds = Dataset(Domain(Variable("state", 2)))
        angle = np.linspace(-np.pi, np.pi, 100)
        velocity = np.linspace(-8., 8., 100)
        X, Y = np.meshgrid(angle, velocity)
        x = X.reshape(-1, 1)
        y = Y.reshape(-1, 1)
        states = np.concatenate([x, y], axis=1)

        ds.notify_batch(state=states)
        mc_pendulum = MC2DPendulum(policy, ds, max_episodes_length=1000, pendulum=pendulum)
        estimate = mc_pendulum.get_v_dataset(1)

        PolicyEvaluation.__init__(self, critic_class, rl_task,
                                  dataset=estimate, policy=policy,
                                            verbose=verbose, plot=plot)
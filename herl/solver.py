import numpy as np
import torch

from herl.dataset import RLDataset
from herl.rl_interface import RLTask, RLAgent

# TODO: shall I rename this file into "rl_collector"?


class RLCollector:

    def __init__(self, dataset: RLDataset, rl_task: RLTask, policy: RLAgent, episode_length: int = None):
        """
        Class to collect data from the reinforcement learning task
        :param dataset: Dataset to fill with data
        :param rl_task: Reinforcement learning Task
        :type rl_task: RLTask
        :param policy: The policy
        """
        self.dataset = dataset
        self.rl_task = rl_task
        self.policy = policy
        self.episode_length = episode_length if episode_length is not None else self.rl_task.max_episode_length

    def collect_samples(self, n_samples, gamma_termination=False):
        """
        Collect n_samples. Note that it leaves the last trajectory open (i.e. when you call get_trajectories,
        you won't be able to see the last trajectory).
        :param n_samples: Number of samples you whish to collect
        :param gamma_termination: when enabled, the episode terminates at each step with probability 1 - gamma.
        :return: None
        """
        i_tot_step = 0
        while True:
            i_step = 0
            terminal = False
            state = self.rl_task.reset()
            self.dataset.notify_new_trajectory()
            while i_step < self.episode_length and not terminal:
                row = self.rl_task.step(self.policy(state))
                state = row["next_state"]
                terminal = bool(row["terminal"][0])
                if gamma_termination:
                    if np.random.uniform() < 1. - self.rl_task.gamma:
                        terminal = True
                self.dataset.notify(**row)
                if terminal:
                    pass
                    # TODO: why there is an unused if? should we put a "continue"
                    # print("hello")
                i_step += 1
                i_tot_step += 1
                if i_tot_step >= n_samples:
                    break
            if i_tot_step >= n_samples:
                break

    def collect_rollouts(self, n_rollouts: int,
                         start_state: np.ndarray = None,
                         start_action: np.ndarray = None,
                         gamma_termination: bool = False):
        """
        Collect n_rollouts.
        :param n_rollouts:
        :param start_state: if you want to start from a specific state.
        :param start_action: if you want to start from a specific action.
        :param gamma_termination: when enabled, the episode terminates at each step with probability 1 - gamma.
        :return: None
        """
        for _ in range(n_rollouts):
            i_step = 0
            terminal = False

            self.dataset.notify_new_trajectory()

            if start_state is None:
                state = self.rl_task.reset()
            else:
                state = self.rl_task.reset(start_state)

            if start_action is not None:
                row = self.rl_task.step(start_action)
                self.dataset.notify(**row)
                i_step += 1

            while i_step < self.episode_length and not terminal:
                row = self.rl_task.step(self.policy(state))
                state = row["next_state"]
                terminal = row["terminal"]
                if gamma_termination:
                    if np.random.uniform() < 1. - self.rl_task.gamma:
                        terminal = True
                self.dataset.notify(**row)
                i_step += 1

        self.dataset.notify_end_trajectory_collection()


# TODO: I think I should remove this class
class IRLAlgorithm:
    """
    This interface represent a generic reinforcement algorithm.
    The algorithm should be able to produce actions that the algorithm belives are optimal (greedy), and actions
    which are meant to be explorative. This pattern is common to many RL algorithms such as DQN, DDPG, ...
    """

    def __init__(self):
        pass

    def learn_from_data(self, **kwargs):
        pass

    def get_greedy_action(self, state=None):
        pass

    def get_explorative_action(self, state=None):
        pass


# TODO: this should probably be moved in rl_interface
class RLSolver:

    """
    This class is meant to collect data and execute an algorithm until a certain condition is met.
    """
    def __init__(self, rl_task, rl_algorithm, terminal_condition):
        self.rl_task = rl_task
        self.rl_algorithm = rl_algorithm
        self.terminal_condition = terminal_condition

    def solve(self):
        i = 0
        while not self.terminal_condition(i, self.rl_task, self.rl_algorithm):
            pass


def get_torch_gradient(f: torch.Tensor, policy, return_torch: bool = False):
    """
    Get the gradient of f w.r.t. the policy's parameters.
    :param f: The parametric function.
    :param policy: The policy parameters.
    :param return_torch: if true returns a torch tensor, otherwise a numpy vector.
    :return: the gradient.
    """
    if return_torch:
        policy.zero_grad()
        f.backward()
        return torch.from_numpy(policy.get_gradient())
    else:
        policy.zero_grad()
        f.backward()
        return policy.get_gradient()




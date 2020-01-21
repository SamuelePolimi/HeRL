import numpy as np

from herl.dataset import MLDataset
from herl.rl_interface import RLTask


class RLCollector:

    def __init__(self, dataset, rl_task, policy, episode_length=None):
        """
        Class to collect data from the reinforcement learning task
        :param dataset: Dataset to fill with data
        :type dataset: MLDataset
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
        Collect n_samples
        :param n_samples:
        :type n_samples: int
        :return: None
        """
        i_tot_step = 0
        while True:
            i_step = 0
            terminal = False
            state = self.rl_task.reset()
            while i_step < self.episode_length and not terminal:
                row = self.rl_task.step(self.policy.get_action(state))
                state = row["next_state"]
                terminal = bool(row["terminal"][0])
                if gamma_termination:
                    if np.random.uniform() < 1. - self.rl_task.gamma:
                        terminal = True
                self.dataset.notify(**row)
                if terminal:
                    print("hello")
                i_step += 1
                i_tot_step += 1
                if i_tot_step >= n_samples:
                    break
            if i_tot_step >= n_samples:
                break

    def collect_rollouts(self, n_rollouts, start_state=None, gamma_termination=False):
        """
        Collect n_rollouts
        :param n_rollouts:
        :type n_rollouts: int
        :return: None
        """
        for _ in range(n_rollouts):
            i_step = 0
            terminal = False
            if start_state is None:
                state = self.rl_task.reset()
            else:
                state = self.rl_task.reset(start_state)
            while i_step < self.episode_length and not terminal:
                row = self.rl_task.step(self.policy.get_action(state))
                state = row["next_state"]
                terminal = row["terminal"]
                if gamma_termination:
                    if np.random.uniform() < 1. - self.rl_task.gamma:
                        terminal = True
                self.dataset.notify(**row)
                i_step += 1


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


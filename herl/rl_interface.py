import numpy as np

from herl.dataset import Domain, Variable, MLDataset
from herl.config import np_float


class RLEnvironment:

    def __init__(self, state_dim, action_dim, environment):
        self.env = environment
        self.state_dim = state_dim
        self.action_dim = action_dim

    def step(self, action):
        a = action.reshape(self.action_dim)
        return self.env.step(a)

    def reset(self, state=None):
        if state is None:
            return self.env.reset()
        else:
            return self.env.reset(state)

    def render(self):
        self.env.render()


class RLTask:

    def __init__(self, environment, gamma=0.99, max_episode_length=np.infty, render=False):
        """
        Reinforcement learning task. Defined by the environment and the discount factor.
        :param environment: Environment of the reinforcement learning task.
        :type environment: RLEnvironment
        :param gamma: Discount factor [0, 1]
        :type gamma: fload
        :param max_episode_length: the horizon of the RL problem
        """
        self.environment = environment
        self.gamma = gamma
        self.domain = Domain(Variable("state", self.environment.state_dim),
                             Variable("action", self.environment.action_dim),
                             Variable("reward", 1),
                             Variable("next_state", self.environment.state_dim),
                             Variable("terminal", 1)
                             )
        self.current_state = None
        self.max_episode_length = max_episode_length

        self.tot_interactions = 0
        self.tot_episodes = 0
        self.returns = []
        self._first_reset = True
        self._partial_return = 0
        self._partial_discounted_return = 0
        self._partial_gamma = 1.
        self.discounted_returns = []
        self.render = render

    def reset(self, state=None):
        """
        Bring the environment in its initial state.
        :return: Initial state
        """
        if not self._first_reset:
            self.returns.append(self._partial_return)
            self.discounted_returns.append(self._partial_discounted_return)

            self._partial_return = 0
            self._partial_discounted_return = 0
            self._partial_gamma = 1.
        else:
            self._first_reset = False
            self.tot_episodes += 1

        if state is None:
            self.current_state = self.environment.reset()
        else:
            self.current_state = self.environment.reset(state)
        return self.current_state

    def step(self, action):
        """
        Apply action to the environment
        :param action: action
        :type action: np.ndarray
        :return: dictionary with a row of the dataset
        """
        self.tot_interactions += 1
        s = np.copy(self.current_state)
        s_n, r, t, _ = self.environment.step(action)
        if self.render:
            self.environment.render()
        self.current_state = s_n

        self._partial_return += r
        self._partial_discounted_return += self._partial_gamma * r
        self._partial_gamma *= self.gamma

        return dict(state=s,
                    action=action,
                    reward=np.array([r], dtype=np_float),
                    next_state=s_n,
                    terminal=np.array([t], dtype=np_float))

    def get_empty_dataset(self, n_max_row=int(10E6), validation=0.):
        """
        Create a dataset correspondent to the domain specified.
        :param n_max_row: Maximum number of rows in the dataset
        :type n_max_row: int
        :param validation: The proportion of validation data
        :type validation: float
        :return: Empty dataset of the right format to save RL data from the specified task.
        :rtype: MLDataset
        """
        return MLDataset(self.domain, n_max_row, validation)

    def commit(self):
        self.discounted_returns.append(self._partial_discounted_return)
        self._partial_discounted_return = 0.
        self._partial_gamma = 1.
        self.returns.append(self._partial_return)
        self._partial_return = 0.


class RLAgent:

    def __init__(self):
        pass

    def get_action(self, state):
        pass


class RLAlgorithm:

    def __init__(self, rl_task, policy):
        self.rl_task = rl_task
        self.policy = policy


class Critic(RLAlgorithm):

    def get_V(self, state):
        raise Exception("Not Implemented")

    def get_Q(self, state, action):
        raise Exception("Not Implemented")


class ModelBased(RLAlgorithm):

    def get_R(self, state, action):
        raise Exception("Not Implemented")

    def get_T(self, state, action):
        raise Exception("Not Implemented")


class PolicyGradient(RLAlgorithm):

    def improve(self):
        raise Exception("Not Implemented")
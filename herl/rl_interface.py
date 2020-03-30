import numpy as np

from herl.dataset import Domain, Variable, MLDataset, Dataset
from herl.config import np_float
from gym.spaces import Box, Discrete, Space
#from herl.utils import deprecated


class RLEnvironment:

    def __init__(self, state_space, action_space, environment_creator, settable=False, deterministic=False,
                 init_deterministic=False):
        """

        :param state_space:
        :type state_space: Box
        :param action_space:
        :type action_space: Box
        :param environment_creator:
        :param settable:
        :param deterministic:
        :param init_deterministic:
        """
        self.env = environment_creator()
        self._env_creator = environment_creator
        self.state_space = state_space
        self.state_dim = state_space.shape[0]
        self.action_space = action_space
        self.action_dim = action_space.shape[0]
        self._settable = settable
        self._deterministic = deterministic
        self._init_deterministic = init_deterministic

    def step(self, action):
        a = action.reshape(self.action_dim)
        return self.env.step(a)

    def reset(self, state=None):
        if state is None:
            return self.env.reset()
        else:
            if self._settable:
                return self.env.reset(state)
            else:
                raise Exception("This environment is not settable (state cannot be overwritten).")

    def render(self):
        self.env.render()

    # TODO:  do we need this, or only in RLTask?
    def set_max_episode_steps(self, max_episode_length):
        """

        :param max_episode_length:
        :type max_episode_length: int
        :return:
        """
        self.env._max_episode_steps = max_episode_length

    def is_settable(self):
        return self._settable

    def is_deterministic(self):
        return self._deterministic

    def is_init_deterministic(self):
        return self._init_deterministic

    def copy(self):
       return  RLEnvironment(self.state_space, self.action_space, self._env_creator, self._settable,
                             self._deterministic, self._init_deterministic)

    def get_grid_dataset(self, states, actions=None):
        grid = [np.linspace(self.state_space.low[i], self.state_space.high[i],
                     states[i]) for i in range(self.state_dim)]
        if actions is not None:
            grid += [np.linspace(self.action_space.low[i], self.action_space.high[i],
                     states[i]) for i in range(self.action_dim)]
        matrix = np.meshgrid(*grid)
        ravel_matrix = np.array([m.ravel() for m in matrix]).T
        if actions is None:
            ds = Dataset(Domain(Variable("state", self.state_dim)), n_max_row=ravel_matrix.shape[0])
            ds.notify_batch(state=ravel_matrix)
            return ds
        else:
            ds = Dataset(Domain(Variable("state", self.state_dim), Variable("action", self.action_dim)),
                         n_max_row=ravel_matrix.shape[0])
            ds.notify_batch(state=ravel_matrix[:, :self.state_dim], action=ravel_matrix[:, self.state_dim:])
            return ds


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
            if self.environment.is_settable():
                self.current_state = self.environment.reset(state)
            else:
                raise Exception("The state of the environment cannot be set.")
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

    def episode(self, policy, starting_state=None, starting_action=None):
        """
        Perform an episode
        :param policy:
        :return:
        """
        self.reset(starting_state)
        t = False
        start = True
        j = 0.
        g = 1.
        while not t:
            if start and starting_action is not None:
                ret = self.step(starting_action)
                start = False
            else:
                ret = self.step(policy(self.current_state))
            j += g * ret["reward"]
            g *= self.gamma
            t = ret["terminal"]
        return j


    def get_empty_dataset(self, n_max_row=None, validation=0.):
        """
        Create a dataset correspondent to the domain specified.
        :param n_max_row: Maximum number of rows in the dataset
        :type n_max_row: int
        :param validation: The proportion of validation data
        :type validation: float
        :return: Empty dataset of the right format to save RL data from the specified task.
        :rtype: MLDataset
        """
        if n_max_row is None:
            return MLDataset(self.domain, self.max_episode_length, validation)
        else:
            return MLDataset(self.domain, n_max_row, validation)

    def commit(self):
        self.discounted_returns.append(self._partial_discounted_return)
        self._partial_discounted_return = 0.
        self._partial_gamma = 1.
        self.returns.append(self._partial_return)
        self._partial_return = 0.

    def copy(self):
        """
        Thi method created a "fresh" copy of the Task (with resetted statistics).
        :return:
        """
        return RLTask(environment=self.environment.copy(), gamma=self.gamma, max_episode_length=self.max_episode_length,
               render=self.render)


class RLAgent:

    def __init__(self):
        pass

    def __call__(self, state, differentiable=False):
        pass

    def is_deterministic(self):
        pass

    def get_parameters(self):
        pass

    def set_parameters(self):
        pass

#    @deprecated
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

    def get_return(self):
        raise Exception("Not Implemented")


class ModelBased(RLAlgorithm):

    def get_R(self, state, action):
        raise Exception("Not Implemented")

    def get_T(self, state, action):
        raise Exception("Not Implemented")


class PolicyGradient(RLAlgorithm):

    def improve(self):
        raise Exception("Not Implemented")

    def get_gradient(self):
        raise Exception("Not Implemented")
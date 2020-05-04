import numpy as np
from gym.spaces import Box
from typing import Iterable, Callable

from herl.dataset import Domain, Variable, MLDataset, Dataset
from herl.config import np_type
from herl.clean_code import deprecated
from herl.utils import Printable


class RLEnvironmentDescriptor:

    def __init__(self, state_space, action_space, deterministic, init_deterministic):
        """
        Descriptor of the environment.
        :param state_space: Space of the state (is mostly needed for visualization matter).
        :type state_space: Box
        :param action_space: Space of the action (is mostly needed for visualization matter).
        :param deterministic: Specify whether the transition is deterministic. This is to save computation for
        Monte Carlo approximation. The user can decide to set it to true even when the environment is _approximately_
        deterministic.
        :type deterministic: bool
        :type action_space: Box
        :param init_deterministic: specify whether the initial state is deterministic. This information is used to save
        computation for Monte Carlo approximation.
        """
        self.state_space = state_space
        self.state_dim = state_space.shape[0]
        self.action_space = action_space
        self.action_dim = action_space.shape[0]
        self._deterministic = deterministic
        self._init_deterministic = init_deterministic

    def is_deterministic(self):
        return self._deterministic

    # TODO: this should be removed
    # @ deprecated
    def is_init_deterministic(self):
        return self._init_deterministic


class RLEnvironment(RLEnvironmentDescriptor):

    # TODO: remove "init_deterministic"
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
        RLEnvironmentDescriptor.__init__(self, state_space, action_space, deterministic, init_deterministic)
        if environment_creator is not None:
            self.env = environment_creator()
        else:
            self.env = None
        self._env_creator = environment_creator
        self._settable = settable

    def step(self, action):
        a = action.reshape(self.action_dim)
        return self.env.step(a)

    # TODO: allow only to specify non-None states
    def reset(self, state=None):
        if state is None:
            # TODO: should raise an error here
            return self.env.reset()
        else:
            if self._settable:
                return self.env.reset(state)
            else:
                raise Exception("This environment is not settable (state cannot be overwritten).")

    def render(self):
        self.env.render()

    # TODO: every environment in this library should be "settable"
    def is_settable(self):
        return self._settable

    def copy(self):
       return RLEnvironment(self.state_space, self.action_space, self._env_creator, self._settable,
                             self._deterministic, self._init_deterministic)

    def get_grid_dataset(self, states: Iterable[int], actions: Iterable[int]=None, step: bool=False) -> Dataset:
        """
        It returns a dataset discretized based on number of states or states and actions.
        When step = True, the method will return a dataset with also reward, next states, and terminal infornmation.
        For example
        :param states: The number of discretization of the states per dimension.
        :param actions: The number of discretization for the actions per dimension.
        When set to 'None', the action will not be discretized.
        :param step: This flag, when activated, provides a classical RL dataset, with states, actions, next states,
        and so on. However, when reward, next_states, and terminal information, then we can turn this flag off.
        :return:
        """
        # TODO is 'initial' flag necessary?
        grid = [np.linspace(self.state_space.low[i], self.state_space.high[i],
                     states[i]) for i in range(self.state_dim)]
        if actions is not None:
            grid += [np.linspace(self.action_space.low[i], self.action_space.high[i],
                     actions[i]) for i in range(self.action_dim)]
        matrix = np.meshgrid(*grid)
        ravel_matrix = np.array([m.ravel() for m in matrix]).T
        if actions is None:
            ds = Dataset(Domain(Variable("state", self.state_dim)), n_max_row=ravel_matrix.shape[0])
            ds.notify_batch(state=ravel_matrix)
            return ds
        else:
            if not step:
                ds = Dataset(Domain(Variable("state", self.state_dim), Variable("action", self.action_dim)),
                             n_max_row=ravel_matrix.shape[0])
                ds.notify_batch(state=ravel_matrix[:, :self.state_dim], action=ravel_matrix[:, self.state_dim:])
            else:
                ds = Dataset(Domain(Variable("state", self.state_dim), Variable("action", self.action_dim),
                               Variable("reward", 1), Variable("next_state", self.state_dim), Variable("terminal", 1),
                               Variable("initial", 1)),
                        n_max_row=ravel_matrix.shape[0])
                for state, action in zip(ravel_matrix[:, :self.state_dim], ravel_matrix[:, self.state_dim:]):
                    self.reset(state)
                    s, r, t, i = self.step(action)
                    ds.notify(state=state, action=action, reward=r, next_state=s,
                                    terminal=t, initial=np.array([1.]))
            return ds

    def get_descriptor(self):
        return RLEnvironmentDescriptor(self.action_space, self.state_space, self._deterministic, self._init_deterministic)

    def close(self):
        pass


class StateDistribution:

    def sample(self) -> np.ndarray:
        """
        Return a state (an array of )
        :return:
        """
        raise NotImplementedError()

    def is_deterministic(self) -> bool:
        raise NotImplementedError()


class DeterministicState(StateDistribution):

    def __init__(self, state: np.ndarray):
        self._state = state

    def sample(self) -> np.ndarray:
        return self._state

    def is_deterministic(self) -> bool:
        return True


class StochasticState(StateDistribution):

    def __init__(self, distribution: Callable[[], np.ndarray]):
        self._distribution = distribution

    def sample(self) -> np.ndarray:
        return self._distribution()

    def is_deterministic(self) -> bool:
        return False


class RLTaskDescriptor:

    def __init__(self, environment_descriptor: RLEnvironmentDescriptor,
                 initial_state_distribution: StateDistribution,
                 gamma: float = 0.99,
                 max_episode_length: np.float = np.infty):
        """
        The description of a task.
        :param environment_descriptor: Description of the environment
        :param initial_state_distribution: Initial state distribution
        :param gamma: Discount Factor
        :param max_episode_length: Maximum number of steps
        """
        self.environment_descriptor = environment_descriptor
        self.gamma = gamma
        self.max_episode_length = max_episode_length
        self.initial_state_distribution = initial_state_distribution


class RLTask:

    def __init__(self, environment: RLEnvironment, initial_state_distribution:StateDistribution, gamma=0.99,
                 max_episode_length=np.infty, render=False):
        """
        Reinforcement learning task. Defined by the environment and the discount factor.
        :param environment: Environment of the reinforcement learning task.
        :type environment: RLEnvironment
        :param gamma: Discount factor [0, 1]
        :type gamma: fload
        :param max_episode_length: the horizon of the RL problem
        """
        self.environment = environment
        self.initial_state_distribution = initial_state_distribution
        self.gamma = gamma
        self.domain = Domain(Variable("state", self.environment.state_dim),
                             Variable("action", self.environment.action_dim),
                             Variable("reward", 1),
                             Variable("next_state", self.environment.state_dim),
                             Variable("terminal", 1),
                             Variable("initial", 1)
                             )
        self.current_state = None
        self.max_episode_length = max_episode_length

        self.tot_interactions = 0
        self.tot_episodes = 0
        self.returns = []
        self._first_reset = True
        self._first_step_ep = False
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
            self.current_state = self.environment.reset(self.initial_state_distribution.sample())
        else:
            if self.environment.is_settable():
                self.current_state = self.environment.reset(state)
            else:
                raise Exception("The state of the environment cannot be set.")
        self._first_step_ep = True
        return self.current_state

    def step(self, action):
        """
        Apply action to the environment
        :param action: action
        :type action: np.ndarray
        :return: dictionary with a row of the dataset
        """
        init_state = self._first_step_ep
        self._first_step_ep = False
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
                    reward=np.array([r], dtype=np_type),
                    next_state=s_n,
                    terminal=np.array([t], dtype=np_type),
                    initial=np.array([init_state], dtype=np_type))

    def episode(self, policy, starting_state=None, starting_action=None):
        """
        Perform an episode
        :param policy:
        :return:
        """
        i = 0
        self.reset(starting_state)
        t = False
        start = True
        j = 0.
        g = 1.
        while not t and i < self.max_episode_length:
            i += 1
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
        return RLTask(self.environment.copy(), self.initial_state_distribution, gamma=self.gamma, max_episode_length=self.max_episode_length,
               render=self.render)

    def get_descriptor(self):
        return RLTaskDescriptor(self.environment.get_descriptor(), self.initial_state_distribution, self.gamma,
                                self.max_episode_length)

    def close(self):
        # TODO: commit?
        self.environment.close()


class RLAgent:

    def __init__(self, deterministic=False, symbol=r"\pi"):
        self._deterministic = deterministic
        self.symbol = symbol

    def __call__(self, state, differentiable=False):
        pass

    def is_deterministic(self):
        return self._deterministic

    def get_parameters(self):
        pass

    def set_parameters(self, values):
        pass

    @deprecated
    def get_action(self, state):
        pass


class RLParametricAgent(RLAgent):

    def get_parameters(self) -> np.ndarray:
        raise NotImplemented()

    def set_parameters(self, values: np.ndarray) -> None:
        raise NotImplemented()

    def load(self, path: str) -> None:
        raise NotImplemented()

    def save(self, path: str) -> None:
        raise NotImplemented()


class RLAlgorithm(Printable):

    def __init__(self, name=""):
        Printable.__init__(self, name, False, False)

    def update(self):
        raise NotImplemented()


class Offline(RLAlgorithm):

    def __init__(self, name, task_descriptor, dataset=None):
        """
        This class represents algorithms that works with an off-line dataset (i.e., the dataset is fixed), the algorithm
        should not interact with the environment.
        :param name: Name of the algorithm
        :type name: str
        :param task_descriptor: Descriptor of the task to solve.
        :type task_descriptor: RLTasktDescriptor
        :param dataset:
        :type dataset: Dataset

        """
        RLAlgorithm.__init__(self, name)
        self.task_descriptor = task_descriptor
        self.dataset = dataset

    def set_dataset(self, dataset: Dataset):
        self.dataset = dataset

    def get_dataset(self):
        return self.dataset


class Online(RLAlgorithm):

    def __init__(self, name: str,  task: RLTask):
        """ Online algorithms must interact with the environment specified in the tast in order to collect the data.
        :param task: The task to solve.
        :type task: RLTask
        """
        RLAlgorithm.__init__(self, name)
        self._task = task


class Actor(RLAlgorithm):

    def __init__(self, name: str, policy: RLAgent):
        """
        Actor algorithms must encode explicitly a policy.
        :param policy:
        """
        RLAlgorithm.__init__(self, name)
        self.policy = policy


class Critic(RLAlgorithm):
    """
    Critic algorithms must estimate the value function.
    """

    def get_V(self, state: np.ndarray) -> np.ndarray:
        raise NotImplemented()

    def get_Q(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        raise NotImplemented()

    def get_return(self) -> np.ndarray:
        raise NotImplemented()


class ModelBased(RLAlgorithm):

    """
    Model Based Algorithms should approximate a model of the transition and eventually of the reward.
    """
    def get_R(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        raise NotImplemented()

    def get_T(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        raise NotImplemented()


class PolicyGradient(Actor):

    """
    Policy gradient algorithm must estimate the gradient of the policy, and improve it.
    """

    def improve(self):
        raise NotImplemented()

    def get_gradient(self):
        raise NotImplemented()

import numpy as np
import gym
import time

from herl.rl_interface import RLEnvironment, RLAgent, RLTask, Domain, Variable
from herl.solver import RLCollector
from herl.dataset import Dataset


def env_from_gym(gym_env):
    return RLEnvironment(gym_env.observation_space.shape[0],
                         gym_env.action_space.shape[0],
                         gym_env)


class RandomPolicyPendulum(RLAgent):

    def __init__(self):
        super().__init__()

    def get_action(self, state):
        return np.random.uniform(-2., 2., 1)


class ConstantPolicyPendulum(RLAgent):

    def __init__(self, action=0.):
        super().__init__()
        self._action = action

    def get_action(self, state):
        return np.array([self._action])


class Pendulum2D(RLEnvironment):

    def __init__(self):
        super().__init__(2, 1, gym.make("Pendulum-v0"))

    def convert(self, state):
        return np.array([np.arctan2(state[1], state[0]), state[2]])

    def reset(self, state=None):
        if state is None:
            return self.convert(self.env.reset())
        else:
            self.env.reset()
            self.env.env.state = state
            return state

    def step(self, action):
        a = action.reshape(self.action_dim)
        state, r, t, i = self.env.step(a)
        return self.convert(state), r, t, i


class RLUniformCollectorPendulum:

    def __init__(self, dataset, n_angles, n_velocities, n_actions):
        """
        Class to collect data from the reinforcement learning task
        :param dataset: Dataset to fill with data
        :type dataset: MLDataset
        :param rl_task: Reinforcement learning Task
        :type rl_task: RLTask
        :param policy: The policy
        """
        self.dataset = dataset
        self.rl_task = RLTask(RLEnvironment(3, 1, gym.make("Pendulum-v0")))
        self.angles = np.linspace(-np.pi, np.pi, n_angles)
        self.velocities = np.linspace(-8., 8., n_velocities)
        self.actions = np.linspace(-2., 2., n_actions)

    def collect_samples(self):
        """
        Collect n_samples
        :param n_samples:
        :type n_samples: int
        :return: None
        """
        for angle in self.angles:
            for velocity in self.velocities:
                for action in self.actions:
                    self.rl_task.reset()
                    self.rl_task.environment.env.state = np.array([angle, velocity])
                    self.rl_task.current_state = np.array([np.cos(angle), np.sin(angle), velocity])
                    row = self.rl_task.step(action)
                    self.dataset.notify(**row)


class MC2DPendulum:

    def __init__(self, policy, dataset, gamma=0.95, max_episodes_length=200):
        """
        Class to collect data from the reinforcement learning task
        :param dataset: Dataset to fill with data
        :type dataset: MLDataset
        :param rl_task: Reinforcement learning Task
        :type rl_task: RLTask
        :param policy: The policy
        """
        self.rl_task = RLTask(Pendulum2D(), gamma=gamma, max_episode_length=max_episodes_length)
        self.policy = policy
        self.dataset = dataset

    def get_v_dataset(self, n_episodes):
        data = self.dataset.get_full()
        return_dataset = Dataset(Domain(Variable("state", 2), Variable("value", 1)), self.dataset.real_size)
        states = data["state"]

        discounts = np.ones((self.rl_task.max_episode_length, 1)) * 0.95
        discounts = (np.cumprod(discounts) / 0.95).reshape(-1, 1)
        collector = RLCollector(self.rl_task.get_empty_dataset(self.rl_task.max_episode_length),
                                self.rl_task,
                                self.policy)

        first = True
        for i, s in enumerate(states):
            returns = []
            start = None
            if first:
                start = time.time()
            for _ in range(n_episodes):
                collector.collect_rollouts(1, start_state=s)
                r = collector.dataset.train_ds.get_full()["reward"]
                d_rewards = np.sum(r * discounts[:r.shape[0]])
                returns.append(d_rewards)
            if first:
                print("Expected time: %f" % ((time.time() - start)*self.dataset.real_size))
                first = False
            return_dataset.notify(state=s, value=np.mean(returns))

        return return_dataset

    def get_q_dataset(self, n_episodes):
        pass
import numpy as np
import gym

from herl.rl_interface import RLEnvironment, RLAgent, RLTask


def env_from_gym(gym_env):
    return RLEnvironment(gym_env.observation_space.shape[0],
                         gym_env.action_space.shape[0],
                         gym_env)


class RandomPolicyPendulum(RLAgent):

    def __init__(self):
        super().__init__()

    def get_action(self, state):
        return np.random.uniform(-2., 2., 1)

class Pendulum2D(RLEnvironment):

    def __init__(self):
        super().__init__(2, 1, gym.make("Pendulum-v0"))

    def convert(self, state):
        return np.array([np.arctan2(state[1], state[0]), state[2]])

    def reset(self, state=None):
        if state is None:
            return self.convert(self.env.reset())
        else:
            return self.convert(self.env.reset(state))

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





import numpy as np

from herl.rl_interface import RLEnvironment, RLAgent


def env_from_gym(gym_env):
    return RLEnvironment(gym_env.observation_space.shape[0],
                         gym_env.action_space.shape[0],
                         gym_env)


class RandomPolicyPendulum(RLAgent):

    def __init__(self):
        super().__init__()

    def get_action(self, state):
        return np.random.uniform(-2., 2., 1)

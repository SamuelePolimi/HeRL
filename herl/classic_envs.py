import gym
import numpy as np
from gym.spaces import Box

from herl.rl_interface import RLEnvironment


def env_from_gym(gym_env):
    return RLEnvironment(gym_env.observation_space.shape[0],
                         gym_env.action_space.shape[0],
                         gym_env)


class Pendulum2D(RLEnvironment):

    def __init__(self, initial_state=None):
        init_det = initial_state is not None
        super().__init__(Box(np.array([-np.pi, -8.]), np.array([np.pi, 8.])),
                            Box(np.array([-2.]), np.array([2.])),
                            lambda: gym.make("Pendulum-v0"), settable=True, deterministic=True,
                            init_deterministic=init_det)

        self.initial_state = initial_state

    def convert(self, state):
        return np.array([np.arctan2(state[1], state[0]), state[2]])

    def reset(self, state=None):
        if state is None:
            if self.initial_state is not None:
                self.env.reset()
                self.env.env.state = self.initial_state
                return self.env.env.state
            return self.convert(self.env.reset())
        else:
            self.env.reset()
            self.env.env.state = state
            return state

    def step(self, action):
        a = action.reshape(self.action_space.shape[0])
        state, r, t, i = self.env.step(a)
        return self.convert(state), r, t, i

    def copy(self):
        return Pendulum2D(self.initial_state)
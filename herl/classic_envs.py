import gym
import numpy as np
from gym.spaces import Box

from herl.rl_interface import RLEnvironment


def env_from_gym(gym_env):
    return RLEnvironment(gym_env.observation_space.shape[0],
                         gym_env.action_space.shape[0],
                         gym_env)


class InfoBox(Box):

    def __init__(self, low, high, description=None, symbol=None):
        super().__init__(low, high)
        self.description = description
        self.symbol = symbol


class LQR(RLEnvironment):

    def __init__(self, A, B, Q, R, initial_state=None, state_box=None, action_box=None):
        """
        :LQR Problem
        :param A:
        :param B:
        :param Q:
        :param R:
        :param initial_state: If we define an initial state then we make the environment completely deterministic.
        If not specified, the initial state is drawn from a Normal(0, I)
        :param state_box: It defines the ranges of (the state )x in (-state_box, state_box) for visualization purposes.
        :param action_box: It defines the ranges of (the action) u in (-state_box, state_box) for visualization purposes.
        """
        init_det = initial_state is not None
        self._d = A.shape[0]
        super().__init__(InfoBox(-state_box, state_box,
                                 ["Dimension %d" % i for i in range(self._d)],
                                 [r"$x_{%i}$" % i for i in range(self._d)]),
                         InfoBox(-action_box, action_box,
                                 ["Torque applied to dimension %d" % i for i in range(self._d)],
                                 [r"$u_{%i}$" % i for i in range(self._d)]),
                         None, settable=True, deterministic=True,
                         init_deterministic=init_det)
        self._state_box = state_box
        self._action_box = action_box
        self._A = A
        self._B = B
        self._Q = Q
        self._R = R
        self._x = None
        self.initial_state = initial_state

    def reset(self, state=None):
        if state is None:
            if self.initial_state is not None:
                self._x = self.initial_state
                return self._x
            self._x = np.random.multivariate_normal(np.zeros(self._d), np.eye(self._d))
            return self._x
        else:
            self._x = state
            return state

    def step(self, action):
        x = self._x.copy()
        self._x = self._A @ x + self._B @ action
        r = np.inner(self._Q @ x, x) + np.inner(self._R @ action, action)
        t = False
        return self._x, r, t, None

    def copy(self):
        return LQR(self._A, self._B, self._Q, self._R, self.initial_state, self._state_box, self._action_box)


class Pendulum2D(RLEnvironment):

    def __init__(self, initial_state=None):
        init_det = initial_state is not None
        super().__init__(InfoBox(np.array([-np.pi, -8.]), np.array([np.pi, 8.]),
                                 ["Angle of the pendulum", "Angular Velocity"], [r"$\theta$", r"\dot{\theta}"]),
                            InfoBox(np.array([-2.]), np.array([2.]), ["Torque applied"], ["F"]),
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
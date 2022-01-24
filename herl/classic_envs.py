import gym
import numpy as np
from typing import Union

import torch
from gym.spaces import Box


from herl.rl_interface import RLEnvironment, StochasticState
from herl.utils import _one_hot, _decode_one_hot

def env_from_gym(gym_env):
    return RLEnvironment(gym_env.observation_space, #.shape[0],
                         gym_env.action_space, #.shape[0],
                         gym_env)


class InfoBox(Box):

    def __init__(self, low, high, description=None, symbol=None):
        super().__init__(low, high, dtype=np.float64)       # TODO: Check dtype
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
                                 [r"x_{%i}" % i for i in range(self._d)]),
                         InfoBox(-action_box, action_box,
                                 ["Torque applied to dimension %d" % i for i in range(self._d)],
                                 [r"u_{%i}" % i for i in range(self._d)]),
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
        r = -np.inner(self._Q @ x, x) - np.inner(self._R @ action, action)
        t = False
        return self._x, r, t, None

    def copy(self):
        return LQR(self._A, self._B, self._Q, self._R, self.initial_state, self._state_box, self._action_box)


class Pendulum2D(RLEnvironment):
    """
    Pendulum described with 2 variables: angle (in radiants, where top is 0, and bottom +-pi) and angular velocity.
    Actions range -2 and 2 and they can be seen as torque.
    """
    # TODO: remove initial state
    def __init__(self, initial_state=None):
        init_det = initial_state is not None
        super().__init__(InfoBox(np.array([-np.pi, -8.]), np.array([np.pi, 8.]),
                                 ["Angle of the pendulum", "Angular Velocity"], [r"\theta", r"\dot{\theta}"]),
                            InfoBox(np.array([-2.]), np.array([2.]), ["Torque applied"], ["F"]),
                            lambda: gym.make("Pendulum-v0").env, settable=True, deterministic=True,
                            init_deterministic=init_det)

        self.initial_state = initial_state

    def convert(self, state):
        return np.array([np.arctan2(state[1], state[0]), state[2]])

    def reset(self, state=None):
        if state is None:
            if self.initial_state is not None:
                self.env.reset()
                self.env.state = self.initial_state
                return self.env.env.state
            return self.convert(self.env.reset())
        else:
            self.env.reset()
            self.env.state = state
            return state

    def step(self, action):
        a = action.reshape(self.action_space.shape[0])
        state, r, t, i = self.env.step(a)
        return self.convert(state), r, False, i

    def copy(self):
        return Pendulum2D(self.initial_state)

    def close(self):
        return self.env.close()


class MDPFeatureInterface:

    def __init__(self, n_states, n_actions):
        self._n_states = n_states
        self._n_actions = n_actions
        self.state_box = None  # type:Union[None, InfoBox]
        self.action_box = None  # type:Union[None, InfoBox]

    def codify_state(self, state):
        pass

    def decodify_state(self, state_features):
        pass

    def codify_action(self, action):
        pass

    def decodify_action(self, action_features):
        pass


class OneHotMDPFeatures(MDPFeatureInterface):

    def __init__(self, n_states, n_actions):
        MDPFeatureInterface.__init__(self, n_states, n_actions)
        self.state_box = InfoBox(np.zeros(n_states), np.ones(n_states))
        self.action_box = InfoBox(np.zeros(n_actions), np.ones(n_actions))

    def codify_state(self, state):
        return _one_hot(state, self._n_states)

    def decodify_state(self, state_features):
        return _decode_one_hot(state_features)

    def codify_action(self, action):
        return _one_hot(action, self._n_states)

    def decodify_action(self, action_features):
        return _decode_one_hot(action_features)


class ImaniFeaturesBackup(MDPFeatureInterface):

    def __init__(self):
        MDPFeatureInterface.__init__(self, 4, 2)
        self._state_matrix = np.array([[0, 1, 0],
                                 [0, 0, 1],
                                 [0, 0, 1],
                                 [1, 0, 0]])
        self.state_box = InfoBox(np.zeros(3), np.ones(3))
        self.action_box = InfoBox(np.zeros(2), np.ones(2))

    def codify_state(self, state):
        if hasattr(state, "shape") and len(state.shape) > 0:
            ret = [self._state_matrix[int(x)] for x in state]
        else:
            ret = self._state_matrix[int(state)]
        if type(state) is torch.Tensor:
            return torch.tensor(ret)
        return np.array(ret)

    def decodify_state(self, state_features):
        try:
            if len(state_features.shape) > 1:
                ret = [np.asscalar(np.min(np.where((self._state_matrix == s).all(axis=1)))) for s in state_features]
                return np.array(ret)
            return np.asscalar(np.min(np.where((self._state_matrix == state_features).all(axis=1))))
        except:
            print("shit")

    def codify_action(self, action):
        return _one_hot(action, self._n_states)

    def decodify_action(self, action_features):
        return _decode_one_hot(action_features)


class ImaniFeatures(MDPFeatureInterface):

    def __init__(self):
        MDPFeatureInterface.__init__(self, 4, 2)
        self._state_matrix = np.array([[0], [1], [1], [2]])

    def codify_state(self, state):
        if hasattr(state, "shape") and len(state.shape) > 0:
            ret = [self._state_matrix[int(x)] for x in state]
        else:
            ret = self._state_matrix[int(state)]
        if type(state) is torch.Tensor:
            return torch.tensor(ret)
        return np.array(ret)

    def codify_action(self, action):
        return action


def get_random_mdp_core(n_states, n_actions):
    P = np.random.uniform(size=(n_actions, n_states, n_states))
    R = np.random.uniform(size=(n_actions, n_states))

    for a in range(n_actions):
        for s in range(n_states):
            P[a, s] = P[a, s]/np.sum(P[a, s])

    mu_0 = np.random.uniform(size=(n_states))
    mu_0 = mu_0/np.sum(mu_0)

    return MDPCore(P, R, mu_0)


def get_imani_mdp():
    """
    Section 5.1, Figure 1, https://arxiv.org/pdf/1811.09013.pdf.
    """
    n_actions = 2
    n_states = 4
    P = np.array([
      # Action 0
      [[0, 1, 0, 0],
       [0, 0, 0, 1],
       [0, 0, 0, 1],
       [0, 0, 0, 1]],
      # Action 1
      [[0, 0, 1, 0],
       [0, 0, 0, 1],
       [0, 0, 0, 1],
       [0, 0, 0, 1]]
    ])
    R = np.array(
        # Action 0
        [[0, 2, 0, 0],
        # Action 1
        [0, 0, 1, 0]]
    )

    mu_0 = np.array([1, 0, 0, 0])

    features = ImaniFeatures()

    return MDP(MDPCore(P, R, mu_0)), features


class MDPCore:

    def __init__(self, P, R, mu_0):

        if P.shape[0] != R.shape[0]:
            raise Exception("P.shape[0] must be equal to R.shape[0]")
        n_actions = P.shape[0]

        if P.shape[1] != P.shape[2]:
            raise Exception("P.shape[1] must be equal to P.shape[2]")

        if P.shape[1] != R.shape[1]:
            raise Exception("P.shape[1] must be equal to R.shape[1]")

        n_states = P.shape[1]

        self._n_states = n_states
        self._n_actions = n_actions
        self._ergodic = self.is_ergodic()

        self._P = P
        self._r = R

        self._mu_0 = mu_0

        self._current_state = None
        self._current_distr = None

        self.reset()

    def is_ergodic(self):   # TODO: implement
        return True

    def reset(self, state=None):
        if state is None:
            self._current_state = np.random.choice(range(self._n_states), p=self._mu_0, size=1)
        else:
            self._current_state = state
        return self._current_state

    def step(self, a):
        a_ravel = np.asscalar(a)
        previous_state = np.asscalar(self._current_state)
        p = self._P[a_ravel, previous_state]
        r = self._r[a_ravel, previous_state]
        self._current_state = np.random.choice(range(self._n_states), p=p, size=1)
        return self._current_state, r, False, None

    def copy(self):
        return MDPCore(self._P, self._r, self._mu_0)


# TODO: Remove features from MDP


class MDP(RLEnvironment):

    def __init__(self, mdp_core: MDPCore, encode: Union[None, MDPFeatureInterface]=None):
        # TODO: ergodic not used (yet)
        n_states = mdp_core._n_states
        n_actions = mdp_core._n_actions
        if encode is not None:
            raise Exception("Use3 of features in MDP is deprecated")
            RLEnvironment.__init__(self, encode.state_box, encode.action_box,
                                   lambda: mdp_core,
                                   True, False, False)
        else:
            RLEnvironment.__init__(self, InfoBox(np.array([0]), np.array([n_states-1])),
                                   InfoBox(np.array([0]), np.array([n_actions-1])),
                                   lambda: mdp_core,
                                   True, False, False)
        self._features = encode
        self._featurized = encode is not None

    def is_featurized(self):
        return self._featurized

    def get_features(self):
        return self._features

    def get_initial_state_sampler(self):
        sampler = lambda: np.random.choice(range(self.env._n_states), p=self.env._mu_0, size=1)
        return StochasticState(sampler)

    def get_states(self, featurize=False):
        if featurize:
            return [self._features.codify_state(s) for s in range(self.env._n_states)]
        return range(self.env._n_states)

    def get_actions(self, featurize=False):
        if featurize:
            return [self._features.codify_action(a) for a in range(self.env._n_actions)]
        return range(self.env._n_actions)

    def get_reward_matrix(self):
        return self.env._r

    def get_initial_state_probability(self):
        return self.env._mu_0

    def get_transition_matrix(self):
        return self.env._P

    def reset(self, state=None):
        ret = self.env.reset(state)
        if self._featurized:
            ret = self._features.codify_state(ret)
        return ret

    def reset_dstribution(self):
        self._current_distr = self.env._mu_0

    def step(self, a):
        a_env = a
        if self._featurized:
            a_env = self._features.decodify_action(a)
        n_s, r, t, i = self.env.step(a_env)
        if self._featurized:
            n_s = self._features.codify_state(n_s)
        return n_s, r, t, i

    def copy(self):
        mdp = MDP(MDPCore(self.env._P, self.env._r, self.env._mu_0), self._features)
        mdp.env.reset()
        return mdp


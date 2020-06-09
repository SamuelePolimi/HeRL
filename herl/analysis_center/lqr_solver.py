import numpy as np, torch
from typing import Callable, Tuple, List, Union
import matplotlib.pyplot as plt

from herl.classic_envs import LQR
from herl.actor import LinearGaussianPolicy, LinearPolicy
from herl.rl_interface import RLTask, Online, Critic, PolicyGradient, Actor,\
    RLParametricModel, RLAgent, DeterministicState
from herl.solver import RLCollector
from herl.rl_analysis import MCAnalyzer, MCEstimate, OnlineEstimate
from herl.rl_visualizer import ValueFunctionVisualizer

"""
This code is copied from Mushroom (https://github.com/MushroomRL/mushroom-rl) (dev branch).
As soon as it will be on the master branch, I'll import from mushroom.
"""


def solve_lqr_linear(lqr, max_iterations=100):
    A, B, Q, R, gamma = _parse_lqr(lqr)

    P = np.eye(Q.shape[0])
    K = _compute_riccati_gain(P, A, B, R, gamma)

    it = 0
    while it < max_iterations:
        P = _compute_riccati_rhs(A, B, Q, R, gamma, K, P)
        K = _compute_riccati_gain(P, A, B, R, gamma)

        it += 1

    return K


def compute_lqr_P(lqr, K):
    A, B, Q, R, gamma = _parse_lqr(lqr)

    L, M = _compute_lqr_intermediate_results(K, A, B, Q, R, gamma)

    vec_P = np.linalg.solve(M, L.reshape(-1))

    return vec_P.reshape(Q.shape)


def compute_lqr_V(x, lqr, K):
    P = compute_lqr_P(lqr, K)
    return -x.T @ P @ x


def check_statbility(A, B, K):
    return np.max(np.abs(np.linalg.eigvals(A + B @ K))) < 1.

def study_properties(lqr_task: RLTask, policy: LinearGaussianPolicy):
    lqr = lqr_task.environment
    K = np.diag(policy.get_parameters())
    A, B = lqr._A, lqr._B
    print("Maximum eigen_vector", np.max(np.abs(np.linalg.eigvals(A + B @ K))))
    da = lqr_task.get_empty_dataset(n_max_row=lqr_task.max_episode_length*100)
    collector = RLCollector(da, lqr_task, policy)
    collector.collect_rollouts(100)
    trajectory_list = da.get_trajectory_list()
    estimated_return = OnlineEstimate()
    estimated_last_state = OnlineEstimate()
    discount = np.cumprod(lqr_task.gamma * np.ones(lqr_task.max_episode_length))/lqr_task.gamma
    for trajectory in trajectory_list[0]:
        t = trajectory
        r = np.sum(t["reward"].ravel()*discount)
        estimated_return.notify(r)
        estimated_last_state.notify(np.linalg.norm(t["state"][-1]))
    print("Mean return ", estimated_return.get_mean())
    print("Std return ", np.sqrt(estimated_return.get_variance()))
    print("Mean last_state ", estimated_last_state.get_mean())
    print("Std leas state ", np.sqrt(estimated_last_state.get_variance()))



def compute_lqg_V(x, lqr, K, Sigma=None):
    P = compute_lqr_P(lqr, -K)
    A, B, Q, R, gamma = _parse_lqr(lqr)
    if not check_statbility(A, B, K):
        raise Exception("""The policy 
        K=%s
        is not stable for 
        A=%s,
        B=%s""" % (K, A, B))
    ret = np.einsum('ij,ji->i', x @ P, x.T)
    stochastic_penalization = 0. if Sigma is None else np.trace(Sigma @ (R + gamma*B.T @ P @ B)) / (1.0 - gamma)
    return - ret - stochastic_penalization


def compute_lqg_gradient(x, lqr, K, Sigma=None):
    A, B, Q, R, gamma = _parse_lqr(lqr)
    L, M = _compute_lqr_intermediate_results(-K, A, B, Q, R, gamma)

    Minv = np.linalg.inv(M)

    n_elems = K.shape[0]*K.shape[1]
    dJ = np.zeros(n_elems)
    for i in range(n_elems):
        dLi, dMi = _compute_lqr_intermediate_results_diff(-K, A, B, R, gamma, i)

        vec_dPi = -Minv @ dMi @ Minv @ L.reshape(-1) + np.linalg.solve(M, dLi.reshape(-1))

        dPi = vec_dPi.reshape(Q.shape)
        stochastic_penalization = 0. if Sigma is None else np.trace(Sigma @ B.T @ dPi @ B)/(1.0-gamma)
        dJ[i] = (x.T @ dPi @ x).item() + gamma*stochastic_penalization

    return -dJ


def _parse_lqr(task: RLTask):
    lqr = task.environment
    if isinstance(lqr, LQR):
        return lqr._A, lqr._B, lqr._Q, lqr._R, task.gamma
    else:
        raise Exception("This parser works only with LQR environment.")


def _compute_riccati_rhs(A, B, Q, R, gamma, K, P):
    return Q + gamma*(A.T @ P @ A - K.T @ B.T @ P @ A - A.T @ P @ B @ K + K.T @ B.T @ P @ B @ K) \
           + K.T @ R @ K


def _compute_riccati_gain(P, A, B, R, gamma):
    return gamma * np.linalg.inv((R + gamma * (B.T @ P @ B))) @ B.T @ P @ A


def _compute_lqr_intermediate_results(K, A, B, Q, R, gamma):
    size = Q.shape[0] ** 2

    L = Q + K.T @ R @ K
    kb = K.T @ B.T
    M = np.eye(size, size) - gamma * (np.kron(A.T, A.T) - np.kron(A.T, kb) - np.kron(kb, A.T) + np.kron(kb, kb))

    return L, M


def _compute_lqr_intermediate_results_diff(K, A, B, R, gamma, i):
    n_elems = K.shape[0]*K.shape[1]
    vec_dKi = np.zeros(n_elems)
    vec_dKi[i] = 1
    dKi = vec_dKi.reshape(K.shape)
    kb = K.T @ B.T
    dkb = dKi.T @ B.T

    dL = dKi.T @ R @ K + K.T @ R @ dKi
    dM = gamma * (np.kron(A.T, dkb) + np.kron(dkb, A.T) - np.kron(dkb, kb) - np.kron(kb, dkb))

    return dL, dM


class LQRAnalyzer(Critic, PolicyGradient, Online):

    def __init__(self, rl_task: RLTask, policy: LinearGaussianPolicy):
        name = "LQR"
        Actor.__init__(self, name, policy)
        Online.__init__(self, name, rl_task)
        self.lqr = rl_task.environment
        self._deterministic = self.policy.is_deterministic()
        self.K = np.diag(self.policy.linear._parameters['weight'].detach().numpy())

    def get_V(self, state: np.ndarray) -> np.ndarray:
        if self._deterministic:
            return compute_lqg_V(state, self._task, self.K)
        else:
            return compute_lqg_V(state, self._task, self.K, self.policy._cov)

    def get_return(self) -> np.ndarray:
        if self._task.get_descriptor().initial_state_distribution.is_deterministic():
            return self.get_V(self._task.get_descriptor().initial_state_distribution.sample())
        else:
            return np.mean([self._task.get_descriptor().initial_state_distribution.sample() for _ in range(10)])

    def get_gradient(self):
        n_params = self.policy.get_parameters().shape[0]
        cov = None if self.policy.is_deterministic() else self.policy._cov
        if self.policy.is_diagonal():
            return -np.diag(compute_lqg_gradient(self._task.get_descriptor().initial_state_distribution.sample(),
                                    self._task, self.K, cov).reshape(n_params, n_params))
        else:
            return -compute_lqg_gradient(self._task.get_descriptor().initial_state_distribution.sample(),
                                    self._task, self.K, cov)


lqr = LQR(A=np.array([[1.2, 0.],
                      [0, 1.1]]),
          B=np.array([[1., 0.],
                      [0., 1.]]),
          R=np.array([[1., 0.],
                      [0., 1.]]),
          Q=np.array([[0.4, 0.],
                      [0., 0.8]]),
          initial_state=np.array([-1., -1.]),
          state_box=np.array([2., 2.]),
          action_box=np.array([2., 2.]))

# initial_state = DeterministicState(np.array([-1., -1.]))
# rl_task = RLTask(lqr, initial_state, gamma=0.9, max_episode_length=100)
# policy = LinearGaussianPolicy(2, 2, covariance=0.00000001*np.eye(2), diagonal=True)
# policy.set_parameters(np.array([-1.1104430687690852, -1.3649958298432607]))
# print(check_statbility(lqr._A, lqr._B, np.diag(np.array([-1.1104430687690852, -1.3649958298432607]))))
# analyzer = LQRAnalyzer(rl_task, policy)
# print(analyzer.get_gradient())
#
# # visualizer = ValueFunctionVisualizer()
# # visualizer.compute(rl_task.environment.get_descriptor(), analyzer, [50, 50])
# # fig, ax = plt.subplots(1, 2)
# # im_1 = visualizer.visualize(ax[0])
# # visualizer.visualize_decorations(ax[0])
#
# mc = MCEstimate(rl_task)
# mc_gradient = MCEstimate(rl_task, max_samples=1000000, absolute_confidence=1E-7)
# analyzer = MCAnalyzer(rl_task, policy, v_mc_estimator=mc, gradient_mc_estimator=mc_gradient, delta_gradient=1E-4)
# # visualizer = ValueFunctionVisualizer()
# # visualizer.compute(rl_task.environment.get_descriptor(), analyzer, [50, 50])
# # im_2 = visualizer.visualize(ax[1])
# # visualizer.visualize_decorations(ax[1])
# #
# # fig.colorbar(im_1, ax=ax[0])
# # fig.colorbar(im_2, ax=ax[1])
#
# print(analyzer.get_gradient())

plt.show()
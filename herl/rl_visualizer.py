import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from multiprocessing.pool import ThreadPool as Pool
from sklearn.neighbors import KernelDensity

from herl.rl_interface import Critic, RLEnvironment
from herl.dataset import Dataset


def plot_value(ax, env, critic, discretization=None, **graphic_args):
    """

    :param ax:
    :type ax: plt.Axes
    :param env: Environment
    :type env: RLEnvironment
    :param critic:
    :type critic: Critic
    :return:
    """

    if env.state_space.shape[0] == 1:
        states = np.linspace(env.state_space.low[0], env.state_space.high[0], discretization[0])
        pool = Pool(mp.cpu_count())
        results = pool.map(critic.get_V, states.reshape(-1, 1))
        ax.plot(states, results, **graphic_args)
    elif env.state_space.shape[0] == 2:
        dataset = env.get_grid_dataset(discretization)
        pool = Pool(mp.cpu_count())
        states = dataset.get_full()["state"]
        results = pool.map(critic.get_V, states)
        shape = [discretization[0], discretization[1]]
        Z = np.array(results).reshape(*shape)
        ax.pcolormesh(states[:, 0].reshape(*shape),
                      states[:, 1].reshape(*shape),
                      Z, **graphic_args)
    else:
        raise Exception("It is not possible to render an environment with state dimension greater than two.")


def plot_q_value(ax, env, critic, discretization=None, **graphic_args):
    """

    :param ax:
    :type ax: plt.Axes
    :param env: Environment
    :type env: RLEnvironment
    :param critic:
    :type critic: Critic
    :return:
    """

    if env.state_space.shape[0] == 1 and env.action_space.shape[0] == 1:
        dataset = env.get_grid_dataset(discretization[0:1], discretization[1:])
        pool = Pool(mp.cpu_count())
        ds = dataset.get_full()
        states, actions = ds["state"], ds["action"]
        evaluate = lambda x: critic.get_Q(x[0], x[1])
        results = pool.map(evaluate, zip(states, actions))
        shape = [discretization[0], discretization[1]]
        Z = np.array(results).reshape(*shape)
        ax.pcolormesh(states.reshape(*shape),
                      actions.reshape(*shape),
                      Z, **graphic_args)
    else:
        raise Exception("It is not possible to render an environment with total space (state + action) greater than two.")


def plot_return(ax, env, critic, low, high, discretization=None, **graphic_args):
    """

    :param ax:
    :type ax: plt.Axes
    :param env: Environment
    :type env: RLEnvironment
    :param critic:
    :type critic: Critic
    :return:
    """

    policy = critic.policy
    if low.shape[0] == 1:
        params = np.linspace(low[0], high[0], discretization[0])
        y = []
        for param in params.reshape(-1, 1):
            policy.set_parameters(param)
            y.append(np.asscalar(critic.get_return()))
        ax.plot(params, y, **graphic_args)
    elif env.state_space.shape[0] == 2:
        x = np.linspace(low[0], high[0], discretization[0])
        y = np.linspace(low[0], high[0], discretization[0])
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                param = np.array([X[i, j], Y[i, j]])
                policy.set_parameters(param)
                Z[i, j] = np.asscalar(critic.get_return())
        ax.pcolormesh(X, Y, Z, **graphic_args)
    else:
        raise Exception("It is not possible to render an environment with state dimension greater than two.")


def plot_state_cloud(ax, dataset, **graphic_args):
    """

    :param dataset: The dataset we want to inspect. Must contain a variable "state"
    :type dataset: Dataset
    :return:
    """
    if dataset.domain.get_variable("state").length == 1:
        states = dataset.get_full()["state"]
        x = states[:, 0]
        ax.plot(x, np.zeros_like(x), **graphic_args)
    elif dataset.domain.get_variable("state").length ==2:
        states = dataset.get_full()["state"]
        ax.scatter(states[:, 0], states[:, 1], **graphic_args)


def plot_state_distribution(ax, environment, dataset, bandwidth=0.1, discretization=np.array([50]), **graphic_args):
    """
    :type environment: RLEnvironment
    :param dataset: The dataset we want to inspect. Must contain a variable "state"
    :type dataset: Dataset
    :return:
    """
    if dataset.domain.get_variable("state").length == 1:
        states = dataset.get_full()["state"]
        kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(states)
        x_lin = np.linspace(environment.state_space.low[0], environment.state_space.high[0], discretization[0])
        y = kde.score_samples(x_lin.reshape(-1, 1))
        ax.plot(x_lin.ravel(), y.ravel(), **graphic_args)
    elif dataset.domain.get_variable("state").length == 2:
        states = dataset.get_full()["state"]
        kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(states)
        x = np.linspace(environment.state_space.low[0], environment.state_space.high[0], discretization[0])
        y = np.linspace(environment.state_space.low[1], environment.state_space.high[1], discretization[1])
        X, Y = np.meshgrid(x, y)
        base = np.array([X.ravel(), Y.ravel()]).T
        z = kde.score_samples(base)
        ax.pcolormesh(X, Y, z.reshape(discretization[0], discretization[1]), **graphic_args)




import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from multiprocessing.pool import ThreadPool as Pool

from herl.rl_interface import Critic, RLEnvironment


def plot_value(ax, env, critic, discretization=None):
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
        results = pool.map(critic.get_V, states)
        ax.plot(states, results)
    elif env.state_space.shape[0] == 2:
        dataset = env.get_grid_dataset(discretization)
        pool = Pool(mp.cpu_count())
        states = dataset.get_full()["state"]
        results = pool.map(critic.get_V, states)
        shape = [discretization[0], discretization[1]]
        Z = np.array(results).reshape(*shape)
        ax.pcolormesh(states[:, 0].reshape(*shape),
                      states[:, 1].reshape(*shape),
                      Z)
    else:
        raise Exception("It is not possible to render an environment with state dimension greater than two.")






import numpy as np
import matplotlib.pyplot as plt

from herl.classic_envs import Pendulum2D
from herl.rl_interface import RLTask, Online, Critic, PolicyGradient
from herl.rl_analysis import MCAnalyzer
from herl.rl_visualizer import plot_value, plot_state_cloud, plot_state_distribution
from herl.actor import UniformPolicy, ConstantPolicy
from herl.solver import RLCollector

env = Pendulum2D(initial_state=np.array([0., 0.]))
task = RLTask(env, max_episode_length=200)

policy = ConstantPolicy([0.])

analizer = MCAnalyzer(task, policy)

# ax = plt.subplot()
# plot_value(ax, env, analizer, discretization=np.array([50, 50]))
# ax.set_title("A")
# plt.show()

uniform_policy = UniformPolicy(env.action_space.low, env.action_space.high)
dataset = task.get_empty_dataset(n_max_row=2000)
collector = RLCollector(dataset, task, uniform_policy)
collector.collect_rollouts(10)
val = dataset.train_ds.where(initial=lambda x: x==1.)["initial"]

ax = plt.subplot()
plot_state_distribution(ax, task.environment, dataset.train_ds, discretization=np.array([50, 50]))
plot_state_cloud(ax, dataset.train_ds)
plt.show()

ax = plt.subplot()
plot_value(ax, task.environment, analizer, discretization=np.array([50, 50]))
plt.show()


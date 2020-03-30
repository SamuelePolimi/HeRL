import numpy as np
import matplotlib.pyplot as plt

from herl.classic_envs import Pendulum2D
from herl.rl_interface import RLTask
from herl.rl_analysis import MCAnalyzer
from herl.utils import ConstantPolicyPendulum
from herl.rl_visualizer import plot_value

env = Pendulum2D(initial_state=np.array([0., 0.]))
task = RLTask(env, max_episode_length=200)

policy = ConstantPolicyPendulum()

analizer = MCAnalyzer(task, policy)

ax = plt.subplot()
plot_value(ax, env, analizer, discretization=np.array([50, 50]))
plt.show()


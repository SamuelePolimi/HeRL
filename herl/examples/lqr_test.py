import numpy as np
import matplotlib.pyplot as plt

from herl.classic_envs import Pendulum2D, LQR
from herl.actor import ConstantPolicy, LinearPolicy
from herl.rl_interface import RLTask
from herl.rl_analysis import MCAnalyzer
from herl.rl_visualizer import plot_value, plot_return, plot_q_value

# Dimension of LQR
d = 1

env = LQR(1.1 * np.eye(d),
          np.eye(d),
          -np.ones((d, d)),
          -np.ones((d, d)),
          np.ones(d),
          -10. * np.ones(d),
          10. * np.ones(d),
          )

task = RLTask(env, gamma=0.9, max_episode_length=200)

policy = LinearPolicy(d, d)

analizer = MCAnalyzer(task, policy)

print(analizer.get_gradient())
ax = plt.subplot()
plot_return(ax, env, analizer, low=np.array([-2.]), high=np.array([0.]), discretization=np.array([50]))
plt.show()

ax = plt.subplot()
plot_q_value(ax, env, analizer, discretization=np.array([50, 50]))
plt.show()
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)

from herl.classic_envs import Pendulum2D, LQR
from herl.actor import ConstantPolicy, LinearPolicy
from herl.rl_interface import RLTask
from herl.rl_analysis import MCAnalyzer
from herl.rl_visualizer import plot_value, plot_return, plot_q_value, plot_gradient_row, plot_gradient

# Dimension of LQR
d = 2

env = LQR(A=np.diag([1.1, 1.2]),
          B=np.diag([1., 1.]),
          Q=np.diag([-0.5, -0.25]),
          R=np.diag([-0.01, -0.01]),
          initial_state=np.ones(d),
          state_box=-10. * np.ones(d),
          action_box=10. * np.ones(d),
          )

task = RLTask(env, gamma=0.9, max_episode_length=100)
factor = 1.
policy = LinearPolicy(d, d)
policy.set_parameters(np.diag([-1.4, -1.4]).ravel())
analizer = MCAnalyzer(task, policy)

print(analizer.get_gradient())

ax = plt.subplot()
fig = plot_return(ax, analizer, analizer.policy, indexes=[0, 3], low=np.array([-1.5, -1.5]), high=np.array([-0.5, -0.5]),
            discretization=np.array([50, 50]))
plt.colorbar(fig)

plot_gradient(ax, analizer, indexes=[0, 3], scale=0.5, length_includes_head=True, head_width=0.01)
plt.show()

fig, axs = plt.subplots(1, 4)
indx = [0, 1, 2, 3]

plot_gradient_row(axs, analizer, indx)

plt.show()
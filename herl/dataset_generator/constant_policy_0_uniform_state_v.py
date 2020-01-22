import gym
import numpy as np
import matplotlib.pyplot as plt
import os

from herl.dataset import Dataset, MLDataset
from herl.rl_interface import RLTask, Domain, Variable
from herl.utils import env_from_gym, MC2DPendulum, RandomPolicyPendulum, ConstantPolicyPendulum

n_angles = 100
n_velocity = 100
n_episodes = 1

ds = Dataset(Domain(Variable("state", 2)))
angle = np.linspace(-np.pi, np.pi, n_angles)
velocity = np.linspace(-8., 8., n_velocity)
X, Y = np.meshgrid(angle, velocity)
x = X.reshape(-1, 1)
y = Y.reshape(-1, 1)
states = np.concatenate([x, y], axis=1)

ds.notify_batch(state=states)

mc_pendulum = MC2DPendulum(ConstantPolicyPendulum(), ds)
estimate = mc_pendulum.get_v_dataset(n_episodes)
data = estimate.get_full()

X = data["state"][:, 0].reshape(n_angles, n_velocity)
Y = data["state"][:, 1].reshape(n_angles, n_velocity)
V = data["value"].reshape(n_angles, n_velocity)


plt.pcolormesh(X, Y, V)
plt.colorbar()
plt.show()

full_path = os.path.realpath(__file__)
path, filename = os.path.split(full_path)

estimate.save(path + "/../datasets/pendulum2d/constant_policy_0_uniform_state_v.npz")

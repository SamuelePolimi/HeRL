import gym
import numpy as np
import matplotlib.pyplot as plt
import os

from herl.dataset import Dataset, MLDataset
from herl.rl_interface import RLTask, Domain, Variable
from herl.utils import env_from_gym, MC2DPendulum, RandomPolicyPendulum, ConstantPolicyPendulum, Pendulum2D
from herl.solver import RLCollector

n_episodes = 25
n_evaluations_episodes = 20


pendulum = Pendulum2D()


rl_task = RLTask(pendulum, max_episode_length=200)

rl_ds = rl_task.get_empty_dataset()
policy = RandomPolicyPendulum()
rl_collector = RLCollector(rl_ds, rl_task, policy)
rl_collector.collect_rollouts(n_episodes)
states = rl_ds.train_ds.get_full()["state"]

plt.scatter(states[:, 0], states[:, 1], s=1)
plt.show()

mc_pendulum = MC2DPendulum(policy, rl_ds.train_ds)

estimate = mc_pendulum.get_v_dataset(n_evaluations_episodes)
data = estimate.get_full()

full_path = os.path.realpath(__file__)
path, filename = os.path.split(full_path)

estimate.save(path + "/../datasets/pendulum2d/uniform_policy_0_state_v.npz")

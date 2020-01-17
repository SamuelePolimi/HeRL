import gym

from herl.dataset import Dataset, MLDataset
from herl.rl_interface import RLTask
from herl.utils import env_from_gym, RandomPolicyPendulum, RLUniformCollectorPendulum

# Create the environment
env = gym.make("Pendulum-v0")
# Make it compatible with our library
env = env_from_gym(env)

# Create the task, by defining the property of the MDP.
# This class will keep track of simple statistics
rl_task = RLTask(env, gamma=0.99, max_episode_length=199)

# This dataset is automatically in the right format for the given task
dataset = rl_task.get_empty_dataset()

# An example of a rl policy just made up for testing purposed
policy = RandomPolicyPendulum()
# this class automatically collects the samples
rl_collector = RLUniformCollectorPendulum(dataset, 10, 10, 2)
# collect four episodes
rl_collector.collect_samples()

rl_task.commit()

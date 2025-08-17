
# import envs and necessary gym packages
from environment.envs.mnist_env import MNIST_ENVIRONMENT
#from gymnasium.envs.registration import register
from gym.envs.registration import register

# register the env using gym's interface
register(
        id = 'MnistEnv-v0',
        entry_point = 'environment.envs.mnist_env:MNIST_ENVIRONMENT',
        max_episode_steps = 1000
)

import gymnasium as gym
#import gym
import numpy as np
np.float = float
import matplotlib.pyplot as plt
import imageio

from evogym.envs import *
import environment.envs as envs
from evogym import get_full_connectivity
from utils import RenderWrapper

class MAKEGIF():

    def __init__(self, args, ind, output_path):
        self.kwargs = vars(args)
        self.ind = ind
        self.output_path = output_path

        # choose an image of a 0 and a 1
        self.zero_image = np.load('data/zero_image.npy')
        self.one_image = np.load('data/one_image.npy')
        self.two_image = np.load('data/two_image.npy')
        self.three_image = np.load('data/three_image.npy')

    def run(self):
        if self.kwargs['task'] == 'MnistEnv-v0':
            imgs = self.run_MnistEnv_v0()
            return imgs
        elif self.kwargs['task'] == 'MnistEnv-v1':
            imgs = self.run_MnistEnv_v1()
            return imgs
        else:
            raise NotImplementedError

    def run_MnistEnv_v0(self):
        body = self.ind.body.to_phenotype()
        connections = get_full_connectivity(body)
        imgs = []
        if 'local' in self.kwargs.keys() and self.kwargs['local']:
            env = gym.make('MnistEnv-v0', body=body, world_json_path='flat_env_1000.json', connections=connections, mnist_data=self.zero_image, is_test=True)
        else:
            env = gym.make('MnistEnv-v0', body=body, world_json_path='flat_env_1000.json', connections=connections, mnist_data=self.zero_image, is_test=False)

        env = RenderWrapper(env, render_mode='img')
        _ = env.reset()

        # run the environment
        fitness_zero = 0
        fitness_one = 0
        for ts in range(1000):
            print(f"ts: {ts}")
            if ts == 500:
                _, reward, done, _, _ = env.step(self.one_image)
            else:
                _, reward, done, _, _ = env.step(None)
            if reward is None:
                return ind, None
            if ts < 500:
                fitness_zero += -1 * reward
            else:
                fitness_one += reward
        imageio.mimsave(f"{self.output_path}_fitness_{self.ind.fitness}_{fitness_zero}_{fitness_one}.gif", env.imgs[0::6], duration=1.0/100)
        imgs.append(env.imgs)
        return imgs

    def run_MnistEnv_v1(self):
        # the way this environment is implemented, makes it uninteresting to gif best individual
        return None

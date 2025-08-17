#from gymnasium import spaces
from gym import spaces
from evogym import EvoWorld
from evogym.envs import EvoGymBase

import sys
sys.path.append('..')

import numpy as np
import os
from copy import deepcopy

class MNIST_ENVIRONMENT(EvoGymBase):
    """ this is a class to generate
    environment with a long flat surface
    and a robot that consists of materials that respond to MNIST digits """

    def __init__(self, body, world_json_path, mnist_data, connections, is_test=False):
        # process the body
        self.body, self.body_to_sim = self.process_body(body)

        # make world
        self.world_length = int(world_json_path.split('_')[2].split('.')[0])
        init_pos = self.world_length // 2 - self.body.shape[1] // 2
        self.world = EvoWorld.from_json(os.path.join('environment/world_data', world_json_path))
        if is_test: # if this is a test environment, we can have the full color
            self.world.add_from_array('robot', self.body_to_sim, init_pos, 2, connections=connections, colors=self.body) # robot is placed at the middle of the world horizontally
        else:
            self.world.add_from_array('robot', self.body_to_sim, init_pos, 2, connections=connections) # robot is placed at the middle of the world horizontally

        # init sim
        EvoGymBase.__init__(self, self.world)

        # set viewer to track objects
        self.default_viewer.track_objects('robot')

        # keep track of time steps
        self.time = 0

        # mnist data
        self.mnist_data = mnist_data

        # open loop acting related variables
        self.actuation_frequency = 50
        self.timestep = 0
        self.sinusoid = np.sin(np.linspace(0, 2*np.pi, self.actuation_frequency))
        self.sinusoid = 0.6 + (self.sinusoid+1) / 2.0
        self.cosine = np.sin(np.linspace(1*np.pi, 3*np.pi, self.actuation_frequency))
        self.cosine = 0.6 + (self.cosine+1) / 2.0

        # specify the action space
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.body[self.body > 2].size,))

        # specify the observation space
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=self.body.shape)

    def process_body(self, body):
        '''
        bodies have extra materials that doesn't actually exist in the simulation
        map these materials to the materials that exist in the simulation

        0 means empty
        original materials \in [1,2,3]
            1, 2 are passive materials
            3 is an active materials controlled by the sinusoidal signal
        rest of the materials are mapped to the 3rd material
            antiphase active material \in [4]
                4 is an active material controlled by the cosine signal
            mechanical passive perception materials \in [5, 6]
                5 responds by shrinking -> mnist 0 to 1 ~ voxel 1.6 to 0.6
                6 responds by expanding -> mnist 0 to 1 ~ voxel 0.6 to 1.6
            in this environment, we only have passive perception materials. 
            these materials sense the corresponding mnist pixel and change their size accordingly.
        '''
        # prepare a body for the simulation
        body_to_sim = deepcopy(body)
        body_to_sim[body_to_sim > 3] = 3
        return body, body_to_sim

    def step(self, action_dict):
        # swap mnist data if new data is provided
        if action_dict['new_mnist_data'] is not None:
            self.mnist_data = action_dict['new_mnist_data']

        # collect pre step information
        pos_1 = self.object_pos_at_time(self.get_time(), "robot")

        # assign action for each voxel
        action = np.ones_like(self.body) * -1.0 #for debuggin purposes
        for row in range(self.body.shape[0]):
            for col in range(self.body.shape[1]):
                # muscle materials
                if self.body[row,col] == 3:
                    action[row,col] = self.sinusoid[self.time]
                elif self.body[row,col] == 4:
                    action[row,col] = self.cosine[self.time]
                # sensory materials
                elif self.body[row,col] == 5:
                    action[row,col] = 1.6 - self.mnist_data[row, col]
                    #if self.mnist_data[row, col] > 0.5:
                    #    action[row,col] = 0.6
                    #else:
                    #    action[row,col] = 1.6
                elif self.body[row,col] == 6:
                    action[row,col] = 0.6 + self.mnist_data[row, col]
                    #if self.mnist_data[row, col] > 0.5:
                    #    action[row,col] = 1.6
                    #else:
                    #    action[row,col] = 0.6
        action = action[self.body > 2]
        action = action.flatten()

        # step
        done = super().step({'robot': action})

        # error check unstable simulation
        if done:
            print("SIMULATION UNSTABLE... TERMINATING")
            return None, None, True, {}

        # collect post step information
        pos_2 = self.object_pos_at_time(self.get_time(), "robot")

        # compute reward (change in center of mass)
        com_1 = np.mean(pos_1, 1)
        com_2 = np.mean(pos_2, 1)
        reward = com_2[0] - com_1[0]
            
        # check if the robot is at the edge of the world
        if com_2[0] < 5 or com_2[0] > self.world_length - 5:
            done = True # TODO: currently the value of this variable is ignored. idk what to do in this situation, have a very big environment and hope the robot doesn't reach the edge??

        # observation
        obs = None
        # keep track of the time
        self.time += 1
        self.time %= self.actuation_frequency

        # observation, reward, has simulation met termination conditions, debugging info
        #return obs, reward, done, None, {}
        if action_dict['return_pos']:
            obs = pos_2
            return obs, reward, done, {}
        else:
            return obs, reward, done, {}

    def reset(self, seed=None, options=None):
        print(f"seed: {seed}")
        print(f"options: {options}")
        
        super().reset()

        # reset time
        self.time = 0

        return None

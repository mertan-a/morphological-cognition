import multiprocessing
import gymnasium as gym
#import gym
import numpy as np

from evogym import get_full_connectivity
import environment.envs as envs

def get_sim_pairs(population, **kwargs):
    sim_pairs = []
    for idx, ind in enumerate(population):
        sim_pairs.append( {'ind':ind, 'kwargs':kwargs} )
    return sim_pairs

def simulate_ind(sim_pair):
    # unpack the simulation pair
    ind = sim_pair['ind']
    kwargs = sim_pair['kwargs']
    # check if the individual has fitness already assigned (e.g. from previous subprocess run. sometimes process hangs and does not return, all the population is re-submitted to the queue)
    if ind.fitness is not None:
        return ind, ind.fitness_full

    # otherwise, simulate the individual
    body = ind.body.to_phenotype()
    connections = get_full_connectivity(body)

    if kwargs['task'] == 'MnistEnv-v0':

        # choose an image of a 0 and a 1
        zero_image = np.load('data/zero_image.npy')
        one_image = np.load('data/one_image.npy')

        # run the environment
        env = gym.make('MnistEnv-v0', body=body, world_json_path='flat_env_1000.json', connections=connections, mnist_data=zero_image, is_test=False)
        env.reset()

        fitness_zero = 0
        fitness_one = 0
        for ts in range(1000):
            if ts == 500:
                _, reward, done, _, _ = env.step(one_image)
            else:
                _, reward, done, _, _ = env.step(None)
            if reward is None:
                return ind, None
            if ts < 500:
                fitness_zero += -1 * reward
            else:
                fitness_one += reward

        return ind, (fitness_zero, fitness_one)

    elif kwargs['task'] == 'MnistEnv-v1':
        # v1 is actually v0, but will be treated a little differently

        # choose an image of a 0, 1, 2, and 3
        zero_image = np.load('data/zero_image.npy')
        one_image = np.load('data/one_image.npy')
        two_image = np.load('data/two_image.npy')
        three_image = np.load('data/three_image.npy')

        all_images = [zero_image, one_image, two_image, three_image]
        fitness_values = []

        for i in range(4):
            # get the image
            img = all_images[i]
            # get a new environment
            env = gym.make('MnistEnv-v0', body=body, world_json_path='flat_env_1000.json', connections=connections, mnist_data=img, is_test=False)
            env.reset()

            fitness = 0
            for ts in range(250):
                # img won't change during lifetime
                _, reward, done, _, _ = env.step(None)

                # if the simulation failed
                if reward is None:
                    return ind, None

                fitness += reward

            fitness_values.append(fitness)
            env.close()

        # return the fitness values
        return ind, fitness_values

    else:
        raise NotImplementedError

def simulate_population(population, **kwargs):
    #get the simulator 
    sim_pairs = get_sim_pairs(population, **kwargs)
    # run the simulation
    finished = False
    while not finished:
        with multiprocessing.Pool(processes=len(sim_pairs)) as pool:
            results_f = pool.map_async(simulate_ind, sim_pairs)
            try:
                results = results_f.get(timeout=580)
                finished = True
            except multiprocessing.TimeoutError:
                print('TimeoutError')
                pass
    # assign fitness
    for r in results:
        ind, fitness_full = r
        for i in population:
            if i.self_id == ind.self_id:
                i.fitness_full = fitness_full
                print(f'Individual {i.self_id} has fitness {i.fitness_full}')
                if kwargs['task'] == 'MnistEnv-v0':
                    if fitness_full is None: # if the simulation failed
                        i.fitness = -1000
                        print('Simulation failed')
                    elif fitness_full[0] < 0 and fitness_full[1] < 0: # robot moves right for image zero and left for image one
                        i.fitness = min( np.abs(fitness_full[0]), np.abs(fitness_full[1]) )
                        print('Robot moves right for image zero and left for image one')
                    elif fitness_full[0] > 0 and fitness_full[1] > 0: # robot moves left for image zero and right for image one
                        i.fitness = min(fitness_full)
                        print('Robot moves left for image zero and right for image one')
                    elif (fitness_full[0] < 0 and fitness_full[1] > 0) or (fitness_full[0] > 0 and fitness_full[1] < 0): # robot moves in the same direction for both images, penalize
                        # take the smaller of the absolute values
                        i.fitness = -1.0 * min( np.abs(fitness_full[0]), np.abs(fitness_full[1]) )
                        print('Robot moves in the same direction for both images')
                    else: # robot does not move in at least one of the images, should not happen too often
                        i.fitness = 0.0
                        print('Robot does not move in at least one of the images')

                elif kwargs['task'] == 'MnistEnv-v1':

                    # if the simulation failed
                    if fitness_full is None:
                        i.fitness = -1000
                        i.fitness_full = None
                        i.behavior = 0 # default behavior
                        print('Simulation failed')
                        continue

                    # fitness will be the minimum distance covered in any of the images
                    i.fitness = min( [abs(f) for f in fitness_full] )

                    # determine individual's behavior
                    if np.sum(np.array(fitness_full) >= 0) == 4 or np.sum(np.array(fitness_full) < 0) == 4: # classifies no digit
                        print('Robot moves in the same direction for all images')
                        i.behavior = 0 # classifies no digit

                    elif np.sum(np.array(fitness_full) < 0) == 3 or np.sum(np.array(fitness_full) >= 0) == 3: # classifies single digit
                        print('Robot moves in the same direction for three images')

                        if (fitness_full[0] >= 0 and fitness_full[1] >= 0 and fitness_full[2] >= 0 and fitness_full[3] < 0) \
                           or \
                           (fitness_full[0] < 0 and fitness_full[1] < 0 and fitness_full[2] < 0 and fitness_full[3] >= 0):
                            i.behavior = 1 # classifies digit 3
                        elif (fitness_full[0] >= 0 and fitness_full[1] >= 0 and fitness_full[2] < 0 and fitness_full[3] >= 0) \
                             or \
                             (fitness_full[0] < 0 and fitness_full[1] < 0 and fitness_full[2] >= 0 and fitness_full[3] < 0):
                            i.behavior = 2 # classifies digit 2
                        elif (fitness_full[0] >= 0 and fitness_full[1] < 0 and fitness_full[2] >= 0 and fitness_full[3] >= 0) \
                             or \
                             (fitness_full[0] < 0 and fitness_full[1] >= 0 and fitness_full[2] < 0 and fitness_full[3] < 0):
                            i.behavior = 3 # classifies digit 1
                        elif (fitness_full[0] < 0 and fitness_full[1] >= 0 and fitness_full[2] >= 0 and fitness_full[3] >= 0) \
                             or \
                             (fitness_full[0] >= 0 and fitness_full[1] < 0 and fitness_full[2] < 0 and fitness_full[3] < 0):
                            i.behavior = 4 # classifies digit 0
                        else:
                            assert False, f"Unexpected behavior for 3: {fitness_full}"

                    elif np.sum(np.array(fitness_full) < 0) == 2: # classifies two digits
                        print('Robot moves in the same direction for two images')

                        if (fitness_full[0] >= 0 and fitness_full[1] >= 0 and fitness_full[2] < 0 and fitness_full[3] < 0) \
                           or \
                           (fitness_full[0] < 0 and fitness_full[1] < 0 and fitness_full[2] >= 0 and fitness_full[3] >= 0):
                            i.behavior = 5
                        elif (fitness_full[0] >= 0 and fitness_full[1] < 0 and fitness_full[2] >= 0 and fitness_full[3] < 0) \
                             or \
                             (fitness_full[0] < 0 and fitness_full[1] >= 0 and fitness_full[2] < 0 and fitness_full[3] >= 0):
                            i.behavior = 6
                        elif (fitness_full[0] < 0 and fitness_full[1] >= 0 and fitness_full[2] >= 0 and fitness_full[3] < 0) \
                             or \
                             (fitness_full[0] >= 0 and fitness_full[1] < 0 and fitness_full[2] < 0 and fitness_full[3] >= 0):
                            i.behavior = 7

                    else:
                        assert False, f"Unexpected behavior: {fitness_full}"
                        
                else:
                    raise NotImplementedError





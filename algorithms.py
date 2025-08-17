import os
import random
import _pickle as pickle
import math
import numpy as np
import time
from copy import deepcopy

from utils import get_files_in, create_folder_structure, natural_sort
from simulator import simulate_population
from make_gif import MAKEGIF
import settings


class MAP_ELITES():
    def __init__(self, args, map):
        self.args = args
        self.map = map
        self.best_fitness = None

    def initialize_optimization(self):
        """
        initialize necessary things for MAP-Elites
        """
        # check if we are continuing or starting from scratch
        from_scratch = False
        if os.path.exists(os.path.join(self.args.rundir, 'pickled_population')):
            pickles = get_files_in(os.path.join(
                self.args.rundir, 'pickled_population'))
            if len(pickles) == 0:
                from_scratch = True
        else:
            from_scratch = True

        if from_scratch:
            print('starting from scratch\n')
            create_folder_structure(self.args.rundir)
            self.starting_generation = 1
            self.current_generation = 0
        else:
            print('continuing from previous run\n')
            pickles = natural_sort(pickles, reverse=False)
            path_to_pickle = os.path.join(
                self.args.rundir, 'pickled_population', pickles[-1])
            self.map = self.load_pickled_map(path=path_to_pickle)
            # extract the generation number from the pickle file name
            self.starting_generation = int(
                pickles[-1].split('_')[-1].split('.')[0]) + 1

    def optimize(self):

        self.initialize_optimization()
        # write a file to indicate that the job is running
        with open(self.args.rundir + '/RUNNING', 'w') as f:
            pass

        for gen in range(self.starting_generation, self.args.nr_generations+1):

            # check if the job should be stopped due to time limit
            if self.args.slurm_queue and time.time() - settings.START_TIME > settings.MAX_TIME[self.args.slurm_queue]:
                print('time limit reached, stopping job')
                settings.STOP = True
                break
            print('GENERATION: {}'.format(gen))
            self.do_one_generation(gen)
            self.record_keeping(gen)
            self.pickle_map()

            if gen % self.args.gif_every == 0 or gen == self.args.nr_generations:
                t = MAKEGIF(self.args, self.map.get_best_individual(), os.path.join(self.args.rundir, f'to_record/{gen}'))
                t.run()

    def do_one_generation(self, gen):

        self.current_generation = gen
        print('PRODUCING OFFSPRINGS')
        offsprings = self.produce_offsprings()
        print('EVALUATING POPULATION')
        self.evaluate(offsprings)
        print('SELECTING NEW POPULATION')
        self.select(offsprings)

    def produce_offsprings(self):
        '''produce offsprings from the current population
        '''
        offspring = self.map.produce_offsprings()
        for i in range(self.args.nr_random_individual):
            offspring.append(self.map.get_random_individual())
        return offspring

    def evaluate(self, population):
        '''evaluate the given population
        '''
        # evaluate the unevaluated individuals
        simulate_population(population=population, **vars(self.args))
        print('population evaluated\n')

    def select(self, population):
        """ update the map with the evaluated offsprings
        """
        self.map.update_map(population)

    def pickle_map(self):
        '''pickle the map for later use
        '''
        pickle_dir = os.path.join(self.args.rundir, 'pickled_population')

        pickle_file = os.path.join(pickle_dir, 'generation_{}.pkl'.format(
            self.current_generation))
        with open(pickle_file, 'wb') as f:
            pickle.dump(self.map, f, protocol=-1)

    def load_pickled_map(self, path):
        '''load the population from a pickle file
        '''
        with open(path, 'rb') as f:
            map = pickle.load(f)
        return map

    def record_keeping(self, gen):
        '''writes a summary and saves the best individual'''
        best_ind = self.map.get_best_individual()
        # keep a fitness over time txt
        with open(os.path.join(self.args.rundir, 'evolution_summary', 'fitness_over_time.txt'), 'a') as f:
            f.write('{}\n'.format(best_ind.fitness))
        # write the best individual
        with open(os.path.join(self.args.rundir, 'to_record', 'best.pkl'), 'wb') as f:
            pickle.dump(best_ind, f, protocol=-1)
        # check whether there is an improvement in the best fitness
        if self.best_fitness is None:
            self.best_fitness = best_ind.fitness
        else:
            if best_ind.fitness > self.best_fitness:
                self.best_fitness = best_ind.fitness
        # also save the current map 
        with open(os.path.join(self.args.rundir, 'to_record', 'map.pkl'), 'wb') as f:
            pickle.dump(self.map, f, protocol=-1)
        # print some useful stuf to the screen
        self.map.print_map()




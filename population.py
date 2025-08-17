import operator
import numpy as np

from individual import INDIVIDUAL
from body import CPPN_BODY


class ARCHIVE():
    """A population of individuals to be used with MAP-Elites"""

    def __init__(self, args):
        """Initialize a population of individuals.

        Parameters
        ----------
        args : object
            arguments object

        """
        self.args = args
        self.map = {}
        if self.args.task == 'MnistEnv-v0':
            # define the bins
            self.n_bins_existing_voxels = 7
            self.n_bins_active_voxels = 7
            self.n_bins_perception_voxels = 7
            for i in range(1, self.n_bins_existing_voxels+1):
                for j in range(1, self.n_bins_active_voxels+1):
                    for k in range(1, self.n_bins_perception_voxels+1):
                        self.map[(i,j,k)] = None
        elif self.args.task == 'MnistEnv-v1':
            # define the bins
            # first dimensions are based on body shape
            self.n_bins_existing_voxels = 7
            self.n_bins_active_voxels = 7
            self.n_bins_perception_voxels = 7
            # last dimension is based on behavior
            self.n_bins_behavior = 8
            for i in range(1, self.n_bins_existing_voxels+1):
                for j in range(1, self.n_bins_active_voxels+1):
                    for k in range(1, self.n_bins_perception_voxels+1):
                        for l in range(self.n_bins_behavior):
                            self.map[(i,j,k,l)] = None
        else:
            raise NotImplementedError

    def get_random_individual(self):
        valid = False
        while not valid:
            # body
            body = CPPN_BODY(self.args)
            ind = INDIVIDUAL(body=body)
            if ind.is_valid():
                valid = True
        return ind

    def produce_offsprings(self):
        """Produce offspring from the current map."""
        # check if there are any individuals in the map
        if len(self) == 0:
            init_population = []
            while len(init_population) < self.args.nr_parents:
                init_population.append(self.get_random_individual())
            return init_population
        # choose nr_parents many random keys from the map. make sure that they are not None
        valid_keys = [ k for k in self.map.keys() if self.map[k] is not None ]
        nr_valid_keys = len(valid_keys) if len(valid_keys) < self.args.nr_parents else self.args.nr_parents
        random_keys_idx = np.random.choice(len(valid_keys), size=nr_valid_keys, replace=False)
        # produce offsprings
        offsprings = []
        for key_idx in random_keys_idx:
            key = valid_keys[key_idx]
            offsprings.append(self.map[key].produce_offspring())
        return offsprings

    def __iter__(self):
        """Iterate over the individuals. Use the expression 'for n in population'."""
        individuals = [ ind for ind in self.map.values() if ind is not None ]
        return iter(individuals)

    def __contains__(self, n):
        """Return True if n is a SoftBot in the population, False otherwise. Use the expression 'n in population'."""
        individuals = [ ind for ind in self.map.values() if ind is not None ]
        try:
            return n in individuals
        except TypeError:
            return False

    def __len__(self):
        """Return the number of individuals in the population. Use the expression 'len(population)'."""
        individuals = [ ind for ind in self.map.values() if ind is not None ]
        return len(individuals)

    def __getitem__(self, x, y):
        """Return individual n.  Use the expression 'population[n]'."""
        return self.map[(x,y)]

    def get_best_individual(self):
        """Return the best individual in the population."""
        individuals = [ ind for ind in self.map.values() if ind is not None ]
        return max(individuals, key=operator.attrgetter('fitness'))

    def get_best_fitness(self):
        """Return the best fitness in the population."""
        individuals = [ ind for ind in self.map.values() if ind is not None ]
        return max(individuals, key=operator.attrgetter('fitness')).fitness

    def update_map(self, population):
        """Update the map with the given population."""
        for ind in population:
            # continue if the individual is not valid
            if ind.fitness is None:
                continue
            # determine the bins
            bin = self.determine_bins(ind)
            # update the map
            if self.map[bin] is None:
                self.map[bin] = ind
            else:
                current_fitness = self.map[bin].fitness
                if ind.fitness > current_fitness:
                    self.map[bin] = ind

    def determine_bins(self, ind):
        """Calculate the bin indices for the given individual."""
        # shape based bins
        existing_voxels = ind.body.count_existing_voxels();
        active_voxels = ind.body.count_active_voxels();
        perception_voxels = ind.body.count_perception_voxels();

        nr_voxel_per_bin = (self.args.bounding_box[0] * self.args.bounding_box[1]) // 7
        bin_existing_voxels = (existing_voxels // nr_voxel_per_bin) + 1; bin_existing_voxels -= 1 if bin_existing_voxels == 8 else 0
        bin_active_voxels = (active_voxels // nr_voxel_per_bin) + 1; bin_active_voxels -= 1 if bin_active_voxels == 8 else 0
        bin_perception_voxels = (perception_voxels // nr_voxel_per_bin) + 1; bin_perception_voxels -= 1 if bin_perception_voxels == 8 else 0

        if self.args.task == 'MnistEnv-v0':
            return (bin_existing_voxels, bin_active_voxels, bin_perception_voxels)
        elif self.args.task == 'MnistEnv-v1':
            return (bin_existing_voxels, bin_active_voxels, bin_perception_voxels, ind.behavior)
        else:
            raise NotImplementedError

    def print_map(self):
        """Print some useful information about the map."""
        # print the best fitness in the map
        print("Best fitness in the map: ", self.get_best_individual().fitness)
        # print the occupancy of the map
        print("Occupancy of the map: ", len(self), "/", len(self.map))

    def get_fitnesses(self):
        """return a numpy array of fitnesses of the individuals in the map,
        with a mask to indicate which bins are not empty"""
        if self.args.task == 'MnistEnv-v0':
            fitnesses = np.zeros((self.n_bins_existing_voxels, self.n_bins_active_voxels, self.n_bins_perception_voxels))
            for i in range(1, self.n_bins_existing_voxels+1):
                for j in range(1, self.n_bins_active_voxels+1):
                    for k in range(1, self.n_bins_perception_voxels+1):
                        if self.map[(i,j,k)] is not None:
                            fitnesses[i-1,j-1,k-1] = self.map[(i,j,k)].fitness
                        else:
                            fitnesses[i-1,j-1,k-1] = -9999
        elif self.args.task == 'MnistEnv-v1':
            fitnesses = np.zeros((self.n_bins_existing_voxels, self.n_bins_active_voxels, self.n_bins_perception_voxels, self.n_bins_behavior))
            for i in range(1, self.n_bins_existing_voxels+1):
                for j in range(1, self.n_bins_active_voxels+1):
                    for k in range(1, self.n_bins_perception_voxels+1):
                        for l in range(self.n_bins_behavior):
                            if self.map[(i,j,k,l)] is not None:
                                fitnesses[i-1,j-1,k-1,l] = self.map[(i,j,k,l)].fitness
                            else:
                                fitnesses[i-1,j-1,k-1,l] = -9999
        else:
            raise NotImplementedError
        # mask for non-empty bins
        mask = fitnesses != -9999
        return fitnesses, mask



        



class Params:
    def __init__(self,
                 mutate_percent_pop=0.25,
                 cells=1,
                 samples=1,
                 generations=10,
                 population_size=24,
                 last_population_size=24,
                 mu=10,
                 lambd=24,
                 lastlambd=24,
                 iters=100,
                 genSteps=100,
                 simSteps=100,
                 crossover_probability=0.1,
                 mutation_probability=0.9,
                 bitFlipProb=0.5):

        self.mutate_percent_pop = mutate_percent_pop # Proportion of model to mutate
        self.cells = cells # Number of cells in the simulation
        self.samples = samples # Number of samples in the simulation
        self.generations = generations # Number of generations to run
        self.population_size = population_size # Size of population
        self.last_population_size = last_population_size # Size of the last population
        self.mu = mu # Number of individuals selected
        self.lambd = lambd # Number of children produced
        self.lastlambd = lastlambd # Number of children produced in the last generation
        self.iters = iters # Number of simulations to try in asynchronous mode
        self.genSteps = genSteps # Steps to find steady state with fake data
        self.simSteps = simSteps # Number of steps each individual is run when evaluating
        self.crossover_probability = crossover_probability # Probability of crossing over a particular parent
        self.mutation_probability = mutation_probability # Probability of mutating a particular parent
        self.bitFlipProb = bitFlipProb # Probability of flipping bits inside mutation

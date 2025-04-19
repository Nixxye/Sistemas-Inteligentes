import math
import random
import itertools
from copy import deepcopy

import matplotlib.pyplot as plt
import utils
from solution import GASolution

class GA:
    def __init__(self, coords, slice, mutation_rate, population_size, generations):
        self.coords = coords
        self.slice = slice
        self.mutation_rate = mutation_rate
        self.population_size = population_size
        self.generations = generations
        self.population = []
        self.best_solution = None
        self.n = len(coords)

        self.init_population()
        for i in range(self.generations):
            self.sex()

    # Inicializa a população com soluções aleatórias
    def init_population(self):
        for _ in range(self.population_size):
            path = random.sample(range(self.n), self.n)
            solution = GASolution(path, self.slice)
            print(len(solution.Path)) 
            print(self.n)
            print()
            if len(solution.Path) == self.n:
                solution.Distance = utils.calculate_solution_distance(
                    solution.Path, self.coords
                )
                self.population.append(solution)

        self.best_solution = min(self.population, key=lambda x: x.Distance)

    def sex(self):
        best = sorted(self.population, key=lambda x: x.Distance)[:len(self.population)//2]
        self.population = []

        for i in range(len(best)//2):
            parent_a = best[i]
            parent_b = best[len(best)-i-1]

            child_a, child_b = self.cross_over(parent_a, parent_b)

            if random.random() < self.mutation_rate:
                self.mutation(child_a)
                self.mutation(child_b)

            self.population.append(child_a)
            self.population.append(child_b)

        best_in_population = min(self.population, key=lambda x: x.Distance)
        if best_in_population.Distance < self.best_solution.Distance:
            self.best_solution = best_in_population


    def cross_over(self, solution_a, solution_b):
        inicio = random.randint(0, self.n - 1)
        new_solution_a = list(itertools.islice(itertools.cycle(solution_a.path), inicio, inicio + self.slice))

        last_elem = new_solution_a[-1]
        index = solution_b.path.index(last_elem) + 1
        for i in range(len(solution_b.path)):
            gene = solution_b.path[(index + i) % len(solution_b.path)]
            if gene not in new_solution_a:
                new_solution_a.append(gene)

        new_solution_b = list(itertools.islice(itertools.cycle(solution_b.path), inicio, inicio + self.slice))

        last_elem = new_solution_b[-1]
        index = solution_a.path.index(last_elem) + 1
        for i in range(len(solution_a.path)):
            gene = solution_a.path[(index + i) % len(solution_a.path)]
            if gene not in new_solution_b:
                new_solution_b.append(gene)

        return GASolution(new_solution_a), GASolution(new_solution_b)



    def mutation(self, solution):
        i, j = random.sample(range(self.n), 2)

        solution.Path[i], solution.Path[j] = (
            solution.Path[j],
            solution.Path[i],
        )
        
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
            if len(solution.Path) == self.n:
                solution.Distance = utils.calculate_solution_distance(
                    solution.Path, self.coords
                )
                self.population.append(solution)

        self.best_solution = min(self.population, key=lambda x: x.Distance)

    def sex(self):
        new_population = []
        total_distance = 0
        for s in self.population:
            total_distance += s.Distance

        prob = self.calculate_sex_chance(total_distance)
        
        for i in range(len(self.population)//2):
            parent_a = random.choices(self.population, weights=prob, k=1)[0]
            parent_b = random.choices(self.population, weights=prob, k=1)[0]

            child_a, child_b = self.cross_over(parent_a, parent_b)

            if random.random() < self.mutation_rate:
                self.mutation(child_a)
                self.mutation(child_b)

            if len(child_a.Path) == self.n:
                child_a.Distance = utils.calculate_solution_distance(
                    child_a.Path, self.coords
                )
            if len(child_b.Path) == self.n:
                child_b.Distance = utils.calculate_solution_distance(
                    child_b.Path, self.coords
                )
            new_population.append(child_a)
            new_population.append(child_b)
        
        best_in_population = min(new_population, key=lambda x: x.Distance)
        if best_in_population.Distance < self.best_solution.Distance:
            self.best_solution = best_in_population
        # substitui a população antiga pela noav
        self.population = new_population
            
    def calculate_sex_chance(self, total_distance):
        prob = []
        for solution in self.population:
            solution.chance = 1 - solution.Distance / total_distance
            prob.append(solution.chance)
        return prob


    def cross_over(self, solution_a, solution_b):
        inicio = random.randint(0, self.n - 1)
        new_solution_a_path = list(itertools.islice(itertools.cycle(solution_a.Path), inicio, inicio + self.slice))

        last_elem = new_solution_a_path[-1]
        index = solution_b.Path.index(last_elem) + 1
        for i in range(len(solution_b.Path)):
            gene = solution_b.Path[(index + i) % len(solution_b.Path)]
            if gene not in new_solution_a_path:
                new_solution_a_path.append(gene)

        if len(new_solution_a_path) == self.n:
            distance_a = utils.calculate_solution_distance(
                new_solution_a_path, self.coords
            )
        new_solution_b_path = list(itertools.islice(itertools.cycle(solution_b.Path), inicio, inicio + self.slice))
        last_elem = new_solution_b_path[-1]
        index = solution_a.Path.index(last_elem) + 1
        for i in range(len(solution_a.Path)):
            gene = solution_a.Path[(index + i) % len(solution_a.Path)]
            if gene not in new_solution_b_path:
                new_solution_b_path.append(gene)

        if len(new_solution_b_path) == self.n:
            distance_b = utils.calculate_solution_distance(
                new_solution_b_path, self.coords
            )
        return GASolution(new_solution_a_path, distance_a), GASolution(new_solution_b_path, distance_b)



    def mutation(self, solution):
        i, j = random.sample(range(self.n), 2)

        solution.Path[i], solution.Path[j] = (
            solution.Path[j],
            solution.Path[i],
        )
        
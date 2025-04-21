import math
import random
import itertools
from copy import deepcopy

import matplotlib.pyplot as plt
import utils
from solution import GASolution

class GA:
    def __init__(self, coords, slice, mutation_rate, population_size, generations, elitism_rate=10, cota_rate=0.8, reset_threshold=15, reset_ratio=0.3):
        self.coords = coords
        self.slice = slice
        self.mutation_rate = mutation_rate
        self.population_size = population_size
        self.generations = generations
        self.population = []
        self.best_solution = None
        self.best_in_last_population = None
        self.elitism_rate = elitism_rate
        self.cota_rate = cota_rate
        self.no_improvement_counter = 0
        self.reset_threshold = reset_threshold  # Número de gerações sem melhora antes de resetar parcialmente
        self.reset_ratio = reset_ratio     # Porcentagem da população a ser resetada

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

    def partial_reset(self):
        reset_count = int(self.population_size * self.reset_ratio)
        for _ in range(reset_count):
            path = random.sample(range(self.n), self.n)
            solution = GASolution(path, self.slice)
            if len(solution.Path) == self.n:
                solution.Distance = utils.calculate_solution_distance(solution.Path, self.coords)
                self.population.append(solution)

        # Remove os piores para manter o tamanho
        self.population = sorted(self.population, key=lambda x: x.Distance)[:self.population_size]

    def sex(self):
        new_population = []

        distances = [s.Distance for s in self.population]
        max_dist = max(distances)

        deltas = [max((max_dist - d), 1e-6) ** 7 for d in distances]
        prob = [d for d in deltas]

        # --- Etapa 1: Elitismo - copiar os 10% melhores ---
        if self.elitism_rate > 0:
            elite_count = max(1, len(self.population) // self.elitism_rate)
            elites = sorted(self.population, key=lambda x: x.Distance)[:elite_count]
            new_population.extend(deepcopy(elites))  # deepcopy para evitar alterações nas elites

        # --- Reprodução para preencher o restante da população ---
        while len(new_population) < self.population_size:
            parent_a = random.choices(self.population, weights=prob, k=1)[0]
            population_without_parent_a = [sol for sol in self.population if sol != parent_a]
            prob_without_parent_a = [p for i, p in enumerate(prob) if self.population[i] != parent_a]
            parent_b = random.choices(population_without_parent_a, weights=prob_without_parent_a, k=1)[0]

            if random.random() < self.cota_rate:
                child_a, child_b = self.cross_over_with_cotas(parent_a, parent_b)
            else: 
                child_a, child_b = self.cross_over(parent_a, parent_b)

            if random.random() < self.mutation_rate:
                self.mutation(child_a)
            if random.random() < self.mutation_rate:
                self.mutation(child_b)

            for child in [child_a, child_b]:
                if len(child.Path) == self.n:
                    child.Distance = utils.calculate_solution_distance(child.Path, self.coords)
                    new_population.append(child)
                    if len(new_population) >= self.population_size:
                        break

        # --- Etapa 2: Remover os 10% piores soluções ---
        new_population = sorted(new_population, key=lambda x: x.Distance)
        if self.elitism_rate > 0:
            removal_count = max(1, len(new_population) // self.elitism_rate)
            new_population = new_population[:-removal_count]

        # Atualiza best_solution
        best_in_population = new_population[0]
        melhora = (self.best_solution.Distance - best_in_population.Distance) / self.best_solution.Distance * 100
        # if melhora != 0:
        #     print(f"Melhor solução da geração é : {melhora:.2f}% melhor do que a anterior")

        if best_in_population.Distance < self.best_solution.Distance:
            self.best_solution = best_in_population

        if self.best_in_last_population is None or best_in_population.Distance < self.best_in_last_population.Distance:
            self.best_in_last_population = best_in_population
            self.no_improvement_counter = 0
        else:
            self.no_improvement_counter += 1
        # Reset parcial se não houve melhora por muitas gerações
        if self.no_improvement_counter >= self.reset_threshold:
            # print("Reset parcial da população por estagnação...")
            self.partial_reset()
            self.no_improvement_counter = 0


        # Preenche novamente se precisar completar o tamanho
        while len(new_population) < self.population_size:
            new_population.append(deepcopy(best_in_population))

        self.population = new_population

            

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

    # Crossover da fatia com a menor distância
    def cross_over_with_cotas(self, solution_a, solution_b):
        def best_slice_from_solution(solution):
            best_slice = None
            best_distance = float("inf")

            for i in range(self.n):
                slice_indices = [(i + j) % self.n for j in range(self.slice)]
                slice_path = [solution.Path[idx] for idx in slice_indices]
                dist = utils.calculate_solution_distance(slice_path, self.coords)
                if dist < best_distance:
                    best_distance = dist
                    best_slice = slice_path
            return best_slice

        # Melhor fatia de solution_a
        slice_a = best_slice_from_solution(solution_a)
        # Complementa com o restante de solution_b
        child_a_path = slice_a[:] + [gene for gene in solution_b.Path if gene not in slice_a]
        distance_a = utils.calculate_solution_distance(child_a_path, self.coords)

        # Melhor fatia de solution_b
        slice_b = best_slice_from_solution(solution_b)
        # Complementa com o restante de solution_a
        child_b_path = slice_b[:] + [gene for gene in solution_a.Path if gene not in slice_b]
        distance_b = utils.calculate_solution_distance(child_b_path, self.coords)

        return GASolution(child_a_path, distance_a), GASolution(child_b_path, distance_b)
      


    def mutation(self, solution):
        # Troca dois genes de lugar
        i, j = random.sample(range(self.n), 2)

        solution.Path[i], solution.Path[j] = (
            solution.Path[j],
            solution.Path[i],
        )
        
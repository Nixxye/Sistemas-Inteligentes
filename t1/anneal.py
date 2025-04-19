import math
import random
from copy import deepcopy

import matplotlib.pyplot as plt
import utils
from solution import TSolution


class SimAnneal(object):
    def __init__(
        self,
        coords,
        T=1,
        alpha=0.95,
        stopping_T=0.001,
        stopping_iter=500,
        biggest_length=10,
    ):
        self.coords = coords
        self.n = len(coords)
        self.intT = T
        self.T = T
        self.alpha = alpha
        self.stopping_T = stopping_T
        self.stopping_iter = stopping_iter
        self.best = None
        self.solutions = []
        self.number_of_iterations = 1
        self.biggest_length = biggest_length
        self.current_solution = None
        self.current_solution = TSolution()
        self.current_solution.Path = self.hungry()
        self.current_solution.Distance = utils.calculate_solution_distance(
            self.current_solution.Path, coords
        )

    def swap_vertices(self, solution):
        new_solution = deepcopy(solution)
        antI = self.n - 1
        proxI = 0
        antJ = self.n - 1
        proxJ = 0
        i, j = random.sample(range(self.n), 2)
        if i - 1 > 0:
            antI = i - 1
        if j - 1 > 0:
            antJ = j - 1
        # Verifica se os vértices são intercambiáveis
        # Verificar com o próximo também
        while self.coords[antI][j] > 0 and self.coords[antJ][i] > 0:
            i, j = random.sample(range(len(new_solution.Path)), 2)

        new_solution.Path[i], new_solution.Path[j] = (
            new_solution.Path[j],
            new_solution.Path[i],
        )
        new_solution.Distance = utils.calculate_solution_distance(
            new_solution.Path, self.coords
        )
        # new_solution.Distance = solution.Distance - self.coords[antI][i] - self.coords[antJ][j] + self.coords[antI][j] + self.coords[antJ][i]
        return new_solution

    def acceptance_probability(self, temperature, delta):
        return random.random() < math.exp(delta / temperature)

    def anneal(self):
        iteration = 1
        while iteration < self.stopping_iter and self.T > self.stopping_T:
            # Gerar uma nova solução:
            new_solution = self.swap_vertices(self.current_solution)
            new_solution.Temperature = self.T
            self.solutions.append(new_solution)
            # Se for melhor, aceita que dói menos:
            if self.best is None:
                self.best = new_solution
            if new_solution.Distance < self.current_solution.Distance:
                self.current_solution = new_solution
                if new_solution.Distance < self.best.Distance:
                    self.best = new_solution
            # Se não for melhor, joga um dado:
            elif self.acceptance_probability(
                self.T,
                (self.current_solution.Distance - new_solution.Distance),
            ):
                self.current_solution = new_solution
            # self.T = self.intT * math.exp(-self.alpha * iteration)
            self.T = self.intT * 1/(self.alpha * iteration)
            iteration += 1
        self.number_of_iterations = iteration

    def innit_solution(self):
        None

    def hungry(self):
        solution = TSolution(self.T, self.alpha, None)
        solution.Path.append(0)  # Começa no nó inicial
        column = 0

        while len(solution.Path) != self.n:
            min_edge = None
            row = column  # Define a linha como o último nó visitado
            next_column = None  # Variável auxiliar para armazenar o próximo nó

            for index, col in enumerate(self.coords[row]):
                if (
                    index != row and index not in solution.Path
                ):  # Evita loops e nós já visitados
                    if min_edge is None or col < min_edge:  # Encontra a menor aresta
                        min_edge = col
                        next_column = index  # Guarda o nó correspondente à menor aresta

            if next_column is not None:  # Confirma que um próximo nó foi encontrado
                solution.Path.append(next_column)
                column = next_column  # Atualiza o nó atual para continuar o processo

        if len(solution.Path) == self.n:
            solution.Distance = utils.calculate_solution_distance(
                solution.Path, self.coords
            )
            self.solutions.append(solution)
            # self.best = solution
            return solution.Path

        return "ERROR: solution not found"

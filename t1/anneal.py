import math
import random
from solution import Solution
import matplotlib.pyplot as plt


class SimAnneal(object):
    def __init__(self, coords, T=100, alpha=0.95, stopping_T=1e-2, stopping_iter=100):
        self.coords = coords
        self.n = len(coords)
        self.T = T
        self.alpha = alpha
        self.stopping_T = stopping_T
        self.stopping_iter = stopping_iter
        self.best = None
        self.solutions = []
        print(self.hungry(coords))

    def anneal(self):
        None
    
    def innit_solution(self):
        None
    
    def hungry(self, matrix):
        solution = Solution(self.n)
        solution.Path.append(0)  # Começa no nó inicial
        column = 0
        
        while len(solution.Path) != self.n:
            min_edge = None
            row = column  # Define a linha como o último nó visitado
            next_column = None  # Variável auxiliar para armazenar o próximo nó

            for index, col in enumerate(matrix[row]):  
                if index != row and index not in solution.Path:  # Evita loops e nós já visitados
                    if min_edge is None or col < min_edge:  # Encontra a menor aresta
                        min_edge = col
                        next_column = index  # Guarda o nó correspondente à menor aresta
            
            if next_column is not None:  # Confirma que um próximo nó foi encontrado
                solution.Path.append(next_column)
                column = next_column  # Atualiza o nó atual para continuar o processo

        if len(solution.Path) == self.n:
            self.solutions.append(solution)
            return solution.Path
        
        return "ERROR: solution not found"
                


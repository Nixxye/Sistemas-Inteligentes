import math
import time

import matplotlib.pyplot as plt
import networkx as nx
import utils
from anneal import SimAnneal
from ga import GA
from solution import TSolution
import random

import time

vertices = 50
max_weight = 800
coords = utils.create_graph(vertices, max_weight)
for i in range(10):
    # Tempera Simulada
    alpha = 1e4
    stopping_T = 10e-16
    stopping_iter = 2e4
    T = 10e1
    sa = SimAnneal(
        coords,
        stopping_iter=stopping_iter,
        T=T,
        stopping_T=stopping_T,
        alpha=alpha,
        biggest_length=max_weight,
    )

    start_sa = time.time()
    sa.anneal()
    end_sa = time.time()
    time_sa = end_sa - start_sa

    # Algoritmo Genético
    ga = GA(
        coords,
        slice=int(vertices / 3),
        mutation_rate=0.8,
        population_size=50,
        generations=10000,
        elitism_rate=60,
        cota_rate=0.5,
        reset_threshold=40,
        reset_ratio=0.3,
    )

    start_ga = time.time()
    ga.solution()
    end_ga = time.time()
    time_ga = end_ga - start_ga

    # Resultados
    gulosa = sa.solutions[0].Distance
    genetica = ga.best_solution.Distance
    tempera = sa.best.Distance

    print("-------------------------------------------------")
    print(f"Solução gulosa: {gulosa}")
    print(f"Melhor solução AG: {genetica} (tempo: {time_ga:.4f} s)")
    print(f"Melhor solução Tempera: {tempera} (tempo: {time_sa:.4f} s)")


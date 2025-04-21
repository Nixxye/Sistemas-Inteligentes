import math
import time

import matplotlib.pyplot as plt
import networkx as nx
import utils
from anneal import SimAnneal
from ga import GA
from solution import TSolution
import random

if __name__ == "__main__":
    # for i in range(40, 110, 10):
    #     print("----------------------------------")
    #     print(f"Execução {i}")
    vertices = 50
    max_weight = 800
    coords = utils.create_graph(vertices, max_weight)

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

    sa.anneal()
    for i in range(1, 11, 1):
        print(f"Execução {i}")
        ga = GA(coords, slice=int(vertices/3), mutation_rate=0.6, population_size=200, generations=10000, elitism_rate= 30, cota_rate=0.8, reset_threshold=10, reset_ratio=0.6)
        # for  solution in ga.population:
        #     print(f"Solução", solution.Path, solution.Distance)
        gulosa = sa.solutions[0].Distance
        genetica = ga.best_solution.Distance
        tempera = sa.best.Distance
        print(f"Melhor solução AG: {genetica}")

        print(f"Solução gulosa: {gulosa}")

        print(f"Melhor solução Tempera: {tempera}")
        
        print(f"Melhora da têmpera: {(gulosa - tempera) / gulosa * 100:.2f}%")
        print(f"Melhora AG: {(gulosa - genetica) / gulosa * 100:.2f}%")
    # N = 0
    # media = 0
    # sa = SimAnneal(
    #     coords,
    #     stopping_iter=stopping_iter,
    #     T=T,
    #     stopping_T=stopping_T,
    #     alpha=alpha,
    #     biggest_length=max_weight,
    # )
    # start = time.time()
    # sa.hungry()
    # end = time.time()
    # print(f"Execution time of hungry(): {end - start:.6f} seconds")
    # sa.current_solution = sa.solutions[0]
    # hh = sa.solutions[0]
    # while N < 20:
    #     sa = SimAnneal(
    #         coords,
    #         stopping_iter=stopping_iter,
    #         T=T,
    #         stopping_T=stopping_T,
    #         alpha=alpha,
    #         biggest_length=max_weight,
    #     )
    #     sa.solutions.append(hh)
    #     sa.current_solution = hh

    #     start = time.time()
    #     sa.anneal()
    #     end = time.time()
    #     print(f"Execution time of anneal(): {end - start:.6f} seconds")

    #     print("qt de solutions: ", len(sa.solutions))
    #     print("guloso -------------------------")
    #     print(f"Distance: {hh.Distance}")
    #     print("end guloso -------------------------")
    #     print("melhor -------------------------")
    #     print(
    #         f"Diferença para o guloso: {(sa.best.Distance - hh.Distance) / hh.Distance * 100:.2f}%"
    #     )
    #     print(f"Distance: {sa.best.Distance}")
    #     print(f"Temp: {sa.best.Temperature}")
    #     print("end melhor -------------------------")

    #     N += 1
    #     if printable == 1:
    #         x = range(len(sa.solutions))
    #         y = [sol.Distance for sol in sa.solutions]
    #         colors = ["green" if val == sa.best.Distance else "red" if val < hh.Distance else "blue" for val in y]
    #         plt.scatter(x, y, c=colors, s=10)
    #         plt.xlabel("iteration")
    #         plt.ylabel("distance")
    #         plt.title("Distance X Iteration")
    #         plt.legend()
    #         plt.show()
    #     media += (sa.best.Distance - hh.Distance) / hh.Distance * 100
    # print(f"média em N: {media/N}%")


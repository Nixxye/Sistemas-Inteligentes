import math
import time

import matplotlib.pyplot as plt
import networkx as nx
import utils
from anneal import SimAnneal
from solution import Solution
import random

if __name__ == "__main__":
    vertices = 100
    max_weight = 800
    if input("type 1 to read from csv") == "1":
        coords = utils.read_csv_matrix("coord.csv")
        print("reading from csv")
    else:
        coords = utils.create_graph(vertices, max_weight)
        utils.write_matrix_to_csv(coords, "coord.csv")
    printable = 1 if input("type 1 to plot the results") == "1" else 2

    # # for solution in sa.solutions:
    alpha = 1e4
    stopping_T = 10e-16
    stopping_iter = 10e3
    T = 10e1
    sa = SimAnneal(
        coords,
        stopping_iter=stopping_iter,
        T=T,
        stopping_T=stopping_T,
        alpha=alpha,
        biggest_length=max_weight,
    )
    #     print(f"Path: {solution.Path}, Distance: {solution.Distance}")
    # print(f"Guloso: {sa.solutions[0].Path}, Distance: {sa.solutions[0].Distance}")
    # print(f"{sa.best.Path}, Distance: {sa.best.Distance}")
    N = 0
    media = 0
    sa = SimAnneal(
        coords,
        stopping_iter=stopping_iter,
        T=T,
        stopping_T=stopping_T,
        alpha=alpha,
        biggest_length=max_weight,
    )
    start = time.time()
    sa.hungry()
    end = time.time()
    print(f"Execution time of hungry(): {end - start:.6f} seconds")
    sa.current_solution = sa.solutions[0]
    hh = sa.solutions[0]
    while N < 200:
        sa = SimAnneal(
            coords,
            stopping_iter=stopping_iter,
            T=T,
            stopping_T=stopping_T,
            alpha=alpha,
            biggest_length=max_weight,
        )
        sa.solutions.append(hh)
        sa.current_solution = hh
        # sa.current_solution = Solution()
        # sa.current_solution.Path = list(range(sa.n))
        # random.shuffle(sa.current_solution.Path)
        # sa.current_solution.Distance = utils.calculate_solution_distance(
        #     sa.current_solution.Path, sa.coords
        # )
        start = time.time()
        sa.anneal()
        end = time.time()
        print(f"Execution time of anneal(): {end - start:.6f} seconds")

        print("qt de solutions: ", len(sa.solutions))
        print("guloso -------------------------")
        print(f"Distance: {hh.Distance}")
        print("end guloso -------------------------")
        print("melhor -------------------------")
        print(
            f"Diferença para o guloso: {(sa.best.Distance - hh.Distance) / hh.Distance * 100:.2f}%"
        )
        print(f"Distance: {sa.best.Distance}")
        print(f"Temp: {sa.best.Temperature}")
        print("end melhor -------------------------")
        # print("ultimo -------------------------")
        # print(
        #     f"Diferença para o guloso: {(sa.solutions[len(sa.solutions)-1].Distance - sa.solutions[0].Distance) / sa.solutions[0].Distance * 100:.2f}%"
        # )
        # print(f"Iterações: {sa.number_of_iterations}")
        # print(f"Distance: {sa.solutions[len(sa.solutions)-1].Distance}")
        # print(f"Temp: {sa.solutions[len(sa.solutions)-1].Temperature}")
        # print("end ultimo -------------------------")
        # print("Temperatura no final das iterações: ", sa.T)
        N += 1
        if printable == 1:
            x = range(len(sa.solutions))
            y = [sol.Distance for sol in sa.solutions]
            colors = ["green" if val == sa.best.Distance else "red" if val < hh.Distance else "blue" for val in y]
            plt.scatter(x, y, c=colors, s=10)
            plt.xlabel("iteration")
            plt.ylabel("distance")
            plt.title("Distance X Iteration")
            plt.legend()
            plt.show()
        media += (sa.best.Distance - hh.Distance) / hh.Distance * 100
    print(f"média em N: {media/N}%")

    # edges = utils.create_networkX_edges_from_solution_path(sa.solutions[0].Path, coords)
    # G = nx.DiGraph()
    # G.add_weighted_edges_from(edges)
    # pos = nx.spring_layout(G, seed=42)
    # nx.draw(
    #     G,
    #     pos,
    #     with_labels=True,
    #     node_color="lightblue",
    #     node_size=1500,
    #     arrows=True,
    #     font_size=14,
    #     connectionstyle="arc3,rad=0.2",
    # )

    # edge_labels = nx.get_edge_attributes(G, "weight")
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=12)
    # plt.show()

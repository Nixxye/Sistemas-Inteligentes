from anneal import SimAnneal
import utils
import networkx as nx
import matplotlib.pyplot as plt

if __name__ == "__main__":
    coords = utils.create_graph(50, 4000)
    # # for solution in sa.solutions:
    alpha = 0.8
    stopping_T = 0.001
    sa = SimAnneal(coords, stopping_iter=100000, T=1, stopping_T=stopping_T, alpha=alpha)
    #     print(f"Path: {solution.Path}, Distance: {solution.Distance}")
    # print(f"Guloso: {sa.solutions[0].Path}, Distance: {sa.solutions[0].Distance}")
    # print(f"{sa.best.Path}, Distance: {sa.best.Distance}")
    while sa.solutions[0].Distance <= sa.best.Distance:
        print(f"Iterações: {sa.number_of_iterations}")
        alpha = alpha / 2
        print(f"Alpha: {alpha}")
        print(f"Distance: {sa.best.Distance}")
        print(f"Temp: {sa.best.Temperature}")
        sa = SimAnneal(coords, stopping_iter=100000, T=1, stopping_T=stopping_T, alpha=alpha)
    print("Anneal é melhor")
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
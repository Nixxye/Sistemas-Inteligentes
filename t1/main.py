from anneal import SimAnneal
import utils
import networkx as nx
import matplotlib.pyplot as plt

if __name__ == "__main__":
    vertices = 200
    max_weight = 400
    coords = utils.create_graph(vertices, max_weight)
    # # for solution in sa.solutions:
    alpha = 0.9999
    stopping_T = 1
    stopping_iter = 32000
    T = 100000
    sa = SimAnneal(coords, stopping_iter=stopping_iter, T=T, stopping_T=stopping_T, alpha=alpha, biggest_length=max_weight)
    #     print(f"Path: {solution.Path}, Distance: {solution.Distance}")
    # print(f"Guloso: {sa.solutions[0].Path}, Distance: {sa.solutions[0].Distance}")
    # print(f"{sa.best.Path}, Distance: {sa.best.Distance}")
    N = 100
    while sa.solutions[0].Distance <= sa.best.Distance or N > 0:
        sa = SimAnneal(coords, stopping_iter=stopping_iter, T=T, stopping_T=stopping_T, alpha=alpha, biggest_length=max_weight)
        alpha *= 0.9
        print('qt de solutions: ', len(sa.solutions))
        print("melhor -------------------------")
        print(f"Diferença para o guloso: {(sa.best.Distance - sa.solutions[0].Distance) / sa.solutions[0].Distance * 100:.2f}%")
        print(f"stopping T: {sa.stopping_T}")
        print(f"Iterações: {sa.number_of_iterations}")
        print(f"Alpha: {alpha}")
        print(f"Distance: {sa.best.Distance}")
        print(f"Temp: {sa.best.Temperature}")
        print("end melhor -------------------------")
        print("ultimo -------------------------")
        print(f"Diferença para o guloso: {(sa.solutions[len(sa.solutions)-1].Distance - sa.solutions[0].Distance) / sa.solutions[0].Distance * 100:.2f}%")
        print(f"stopping T: {sa.stopping_T}")
        print(f"Iterações: {sa.number_of_iterations}")
        print(f"Alpha: {alpha}")
        print(f"T: {T}")
        print(f"Distance: {sa.solutions[len(sa.solutions)-1].Distance}")
        print(f"Temp: {sa.solutions[len(sa.solutions)-1].Temperature}")
        print("end ultimo -------------------------")
        N -= 1

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
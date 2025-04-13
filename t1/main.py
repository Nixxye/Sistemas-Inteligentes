from anneal import SimAnneal
import utils
import networkx as nx
import matplotlib.pyplot as plt

if __name__ == "__main__":
    coords = utils.read_csv_matrix("coord.csv")
    sa = SimAnneal(coords, stopping_iter=5000)
    edges = utils.create_networkX_edges_from_matrix(coords)
    G = nx.DiGraph()
    G.add_weighted_edges_from(edges)
    pos = nx.spring_layout(G, seed=42)
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color="lightblue",
        node_size=1500,
        arrows=True,
        font_size=14,
        connectionstyle="arc3,rad=0.2",
    )

    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=12)
    plt.show()
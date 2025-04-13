import random
import pandas as pd

def read_csv_matrix(path):
    return pd.read_csv(path, header=None).values.tolist()

def write_matrix_to_csv(matrix, filename):
    df = pd.DataFrame(matrix)  
    df.to_csv(filename, index=False, header=False) 

def create_graph(vertices: int, max_weight: int):
    return [
        [random.randint(1, max_weight) if row != col else 0 for col in range(vertices)]
        for row in range(vertices)
    ]


def print_matrix(matrix):
    for row in matrix:
        print(row)


def create_networkX_edges_from_matrix(matrix):
    edges = []
    for row in range(len(matrix)):
        for col in range(len(matrix[row])):
            if row != col:
                edges.append((row, col, matrix[row][col]))

    return edges

def create_networkX_edges_from_solution_path(path):
    edges = []
    for vertice in path:
        edges.append()
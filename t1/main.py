import math
import time
import matplotlib.pyplot as plt
import networkx as nx
import utils
from anneal import SimAnneal
from ga import GA
from solution import TSolution
import random
import numpy as np


max_weight = 800
alpha = 1e4
stopping_T = 10e-16
stopping_iter = 2e4
T = 10e1

vertex_range = range(10, 151, 10)
melhoras_ag = []
melhoras_tempera = []

for vertices in vertex_range:
    soma_melhora_ag = 0
    soma_melhora_tempera = 0

    for _ in range(10):
        coords = utils.create_graph(vertices, max_weight)

        # Têmpera Simulada
        sa = SimAnneal(
            coords,
            stopping_iter=stopping_iter,
            T=T,
            stopping_T=stopping_T,
            alpha=alpha,
            biggest_length=max_weight,
        )

        sa.anneal()

        # Algoritmo Genético
        ga = GA(
            coords,
            slice=int(vertices / 3),
            mutation_rate=0.8,
            population_size=30,
            generations=300,
            elitism_rate=60,
            cota_rate=0.5,
            reset_threshold=40,
            reset_ratio=0.3,
            hungry_rate=0.4,
        )

        ga.solution()

        # Resultados
        gulosa = sa.solutions[0].Distance
        genetica = ga.best_solution.Distance
        tempera = sa.best.Distance

        soma_melhora_ag += ((gulosa - genetica) / gulosa) * 100
        soma_melhora_tempera += ((gulosa - tempera) / gulosa) * 100
        if soma_melhora_tempera < 0: 
            soma_melhora_tempera = 0

    media_ag = soma_melhora_ag / 10
    media_tempera = soma_melhora_tempera / 10

    melhoras_ag.append(media_ag)
    melhoras_tempera.append(media_tempera)

    print(f"Vértices: {vertices} | Média melhora AG: {media_ag:.2f}% | Média melhora Têmpera: {media_tempera:.2f}%")

# Plot
plt.figure(figsize=(10, 6))
plt.plot(list(vertex_range), melhoras_ag, marker='o', label='AG (média de melhora %)')
plt.plot(list(vertex_range), melhoras_tempera, marker='s', label='Têmpera Simulada (média de melhora %)')
plt.title('Média de Melhora em Relação à Solução Gulosa')
plt.xlabel('Número de Vértices')
plt.ylabel('Melhora (%)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

max_weight = 800
alpha = 1e4
stopping_T = 10e-16
stopping_iter = 2e4
T = 10e1

vertices = 50
melhoras_ag = []
melhoras_tempera = []

coords = utils.create_graph(vertices, max_weight)
for _ in range(50):

    # Têmpera Simulada
    sa = SimAnneal(
        coords,
        stopping_iter=stopping_iter,
        T=T,
        stopping_T=stopping_T,
        alpha=alpha,
        biggest_length=max_weight,
    )

    sa.anneal()

    # Algoritmo Genético
    ga = GA(
        coords,
        slice=int(vertices / 3),
        mutation_rate=0.8,
        population_size=30,
        generations=300,
        elitism_rate=60,
        cota_rate=0.5,
        reset_threshold=40,
        reset_ratio=0.3,
        hungry_rate=0.4,
    )

    ga.solution()

    # Resultados
    gulosa = sa.solutions[0].Distance
    genetica = ga.best_solution.Distance
    tempera = sa.best.Distance

    melhora_ag = ((gulosa - genetica) / gulosa) * 100
    melhora_tempera = ((gulosa - tempera) / gulosa) * 100
    melhoras_ag.append(melhora_ag)
    melhoras_tempera.append(max(melhora_tempera, 0))  # evita negativos

# Estatísticas
media_ag = np.mean(melhoras_ag)
std_ag = np.std(melhoras_ag)
media_tempera = np.mean(melhoras_tempera)
std_tempera = np.std(melhoras_tempera)

print("------------------------------------------------")
print(f"Resultados para {vertices} vértices (50 execuções):")
print(f"AG - Média: {media_ag:.2f}% | Desvio padrão: {std_ag:.2f}%")
print(f"Têmpera - Média: {media_tempera:.2f}% | Desvio padrão: {std_tempera:.2f}%")

# Boxplot
plt.figure(figsize=(8, 6))
plt.boxplot([melhoras_ag, melhoras_tempera], labels=["AG", "Têmpera"])
plt.title(f"Distribuição das Melhoras sobre a Solução Gulosa ({vertices} vértices)")
plt.ylabel("Melhora (%)")
plt.grid(True)
plt.tight_layout()
plt.show()
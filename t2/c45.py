import math
from collections import Counter
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
import os

MAX_DEPTH = 0
NOS_PODADOS = 0

class c45Node:
    def __init__(self, data, lvl=0, maxDepth=MAX_DEPTH, entropy=None, method='entropy'):
        self.maxDepth = maxDepth
        self.lvl = lvl
        self.children = []
        self.attribute = None
        self.threshold = None
        self.data = data
        self.classification = None
        if entropy is None:
            self.entropy = self.calcEntropy()
        else:
            self.entropy = entropy
        self.method = method

    def count_nodes(self):
        count = 1  # Conta o nó atual
        for child in self.children:
            count += child.count_nodes()
        return count
    
    def all_same_class(self):
        classes = [row[-1] for row in self.data]
        return all(c == classes[0] for c in classes)
    
    def build_tree(self):
        if self.lvl > self.maxDepth or len(self.data) <= 2 or self.all_same_class():
            self.classification = get_majority_class(self.data)
            return
        att = None
        minEntropy = None
        minEntropyThreshold = None
        for i in range(1, len(self.data[0])-2): # Ignora o ID e a classe
            self.data = sorted(self.data, key=lambda x: x[i])
            threshold = self.calcThreshold(i, method=self.method)
            if threshold is None:
                continue
            left = [row for row in self.data if row[i] < threshold]
            right = [row for row in self.data if row[i] >= threshold]
            total = len(self.data)

            entropy1 = self.calcEntropy(left) * len(left) / total
            entropy2 = self.calcEntropy(right) * len(right) / total
            entropy = entropy1 + entropy2
            if minEntropy is None or entropy < minEntropy:
                minEntropy = entropy
                minEntropyThreshold = threshold
                att = i
        if att is None:
            return
        self.attribute = att
        #print(f"Dividindo no atributo {att} com threshold {minEntropyThreshold} (entropia: {minEntropy})")
        self.threshold = minEntropyThreshold
        if len(left) == 0 or len(right) == 0:
            self.classification = get_majority_class(self.data)
            return
        self.separateData(att, minEntropyThreshold, entropy1, entropy2)
        for child in self.children:
            child.build_tree()

    def calcThreshold(self, attribute, method='median'):
        if method == 'median':
            return self.calcMedThreshold(attribute)
        elif method == 'average':
            return self.calcAvgThreshold(attribute)
        elif method == 'entropy':
            return self.calcEntThreshold(attribute)
        else:
            raise ValueError("Método de cálculo de threshold desconhecido: {}".format(method))

    def calcMedThreshold(self, attribute):
        values = [row[attribute] for row in self.data]
        n = len(values)
        if n % 2 == 1:
            return values[n // 2]
        else:
            return (values[n // 2 - 1] + values[n // 2]) / 2

    def calcAvgThreshold(self, attribute):
        values = [row[attribute] for row in self.data]
        return sum(values) / len(values) if values else 0

    def calcEntThreshold(self, attribute):
        best_threshold = None
        best_entropy = float('inf')
        for i in range(len(self.data) - 1):
            class_curr = self.data[i][-1]
            class_next = self.data[i + 1][-1]
            if class_curr != class_next:
                threshold = (self.data[i][attribute] + self.data[i + 1][attribute]) / 2
                left = [x for x in self.data if x[attribute] < threshold]
                right = [x for x in self.data if x[attribute] >= threshold]
                total = len(self.data)
                entropy = (len(left) / total) * self.calcEntropy(left) + (len(right) / total) * self.calcEntropy(right)
                if entropy < best_entropy:
                    best_entropy = entropy
                    best_threshold = threshold
        return best_threshold

    def calcEntropy(self, data=None):
        if data is None:
            data = self.data
        frequency = [0, 0, 0, 0]
        for i in range(len(data)):
            frequency[int(data[i][-1])-1] += 1
        ent = 0
        for x in frequency:
            if x > 0:
                f = x / len(data)
                ent -= f * math.log2(f)
        return ent

    def separateData(self, attribute, threshold, entropy1, entropy2):
        if self.lvl > self.maxDepth:
            return
        left_data = [row for row in self.data if row[attribute] < threshold]
        right_data = [row for row in self.data if row[attribute] >= threshold]
        if len(left_data) == 0 or len(right_data) == 0:
            return
        self.children = [
            c45Node(data=left_data, lvl=self.lvl+1, maxDepth=self.maxDepth, entropy=entropy1, method=self.method),
            c45Node(data=right_data, lvl=self.lvl+1, maxDepth=self.maxDepth, entropy=entropy2, method=self.method)
        ]

    def classify(self, data):
        if self.children == []:
            if self.classification:
                return self.classification
            else:
                class_counts = {}
                for item in self.data:
                    label = item[-1]
                    class_counts[label] = class_counts.get(label, 0) + 1
                return max(class_counts.items(), key=lambda x: x[1])[0]
        else:
            if data[self.attribute] < self.threshold:
                return self.children[0].classify(data)
            else:
                return self.children[1].classify(data)

def validate(data, tree):
    correct = 0
    for i in range(len(data)):
        if int(tree.classify(data[i])) == int(data[i][-1]):
            correct += 1
        #print(f"Classificando {i+1}/{len(data)}: {data[i]} -> {tree.classify(data[i])} (esperado: {data[i][-1]})")
    return correct / len(data)


def pruning(tree, confidence=0.75, z=None):
    global NOS_PODADOS
    if z is None:
        z = norm.ppf((1 + confidence) / 2)  # valor z do intervalo de confiança

    for child in tree.children:
        pruning(child, confidence, z)

    if tree.children:
        majority_class = get_majority_class(tree.data)
        n = len(tree.data)
        f_leaf = sum(1 for row in tree.data if row[-1] != majority_class)
        error_as_leaf = laplace_error(f_leaf, n, z)

        f_subtree = count_subtree_errors(tree)
        error_subtree = laplace_error(f_subtree, n, z)

        if error_as_leaf <= error_subtree:
            tree.children.clear()
            tree.attribute = None
            tree.threshold = None
            tree.classification = majority_class
            NOS_PODADOS += 1


def get_majority_class(data):
    labels = [row[-1] for row in data]
    return Counter(labels).most_common(1)[0][0]

def laplace_error(f, n, z):
    if n == 0:
        return 0
    f_n = f / n
    return (f + z**2 / 2 + z * math.sqrt(f_n * (1 - f_n) + z**2 / (4 * n))) / (n + z**2)

def count_subtree_errors(tree):
    if not tree.children:
        return sum(1 for row in tree.data if row[-1] != tree.classification)
    return sum(count_subtree_errors(child) for child in tree.children)


def classify_row(tree, row):
    while tree.children:
        if row[tree.attribute] < tree.threshold:
            tree = tree.children[0]
        else:
            tree = tree.children[1]
    return tree.classification


def testar_variacoes(data_path="dataset/treino_sinais_vitais_com_label.csv"):
    global NOS_PODADOS
    data = pd.read_csv(data_path).values
    training_percentages = [0.1, 0.3, 0.5, 0.7, 0.9]
    methods = ['median', 'average', 'entropy']
    confidence = 0.69
    maxDepth_values = [1, 3, 6, 9, 12, 15, 18, 21]  # Vários máximos de profundidade para comparar

    os.makedirs("graficos/c45", exist_ok=True)

    # Estrutura: resultados[method][maxDepth]['antes' ou 'depois'][i] = acurácia na i-ésima porcentagem
    resultados = {
        method: {
            maxDepth: {'antes': [], 'depois': []} for maxDepth in maxDepth_values
        }
        for method in methods
    }

    for method in methods:
        for maxDepth in maxDepth_values:
            for perc in training_percentages:
                treino = data[:int(len(data)*perc)]
                teste = data[int(len(data)*perc):]

                split_index = int(len(treino) * 0.7)
                # treino_construcao = treino[:split_index]
                # treino_validacao = treino[split_index:]

                tree = c45Node(treino, maxDepth=maxDepth, method=method)
                tree.build_tree()

                acc_antes = validate(teste, tree)
                NOS_PODADOS = 0
                pruning(tree, confidence=confidence)
                acc_depois = validate(teste, tree)
                print(f"Número de nós podados: {NOS_PODADOS}")
                resultados[method][maxDepth]['antes'].append(acc_antes)
                resultados[method][maxDepth]['depois'].append(acc_depois)

        # Agora plota um gráfico por método com linhas para cada maxDepth
        plt.figure(figsize=(10, 6))

        for maxDepth in maxDepth_values:
            plt.plot(training_percentages, 
                     [x * 100 for x in resultados[method][maxDepth]['antes']], 
                     label=f'Antes poda - Prof. {maxDepth}', linestyle='--')
            plt.plot(training_percentages, 
                     [x * 100 for x in resultados[method][maxDepth]['depois']], 
                     label=f'Depois poda - Prof. {maxDepth}', linestyle='-')

        plt.title(f"Acurácia x Porcentagem de Treino ({method})")
        plt.xlabel("Porcentagem de treino")
        plt.ylabel("Acurácia (%)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"graficos/c45/{method}_acuracia_comparativa.png")
        plt.close()


if __name__ == "__main__":
    testar_variacoes()

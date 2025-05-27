import math
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
import os

MAX_DEPTH = 1
NOS_PODADOS = 0

class c45Node:
    def __init__(self, data, lvl=0, maxDepth=MAX_DEPTH, entropy=None, method='entropy'):
        self.maxDepth = maxDepth
        self.lvl = lvl
        self.children = []
        self.attribute = None
        self.threshold = None
        self.data = data
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
    
    def build_tree(self):
        if self.lvl > MAX_DEPTH or len(self.data) == 0:
            return
        att = None
        minEntropy = None
        minEntropyThreshold = None
        for i in range(len(self.data[0])-1):
            self.data = sorted(self.data, key=lambda x: x[i])
            threshold = self.calcThreshold(i, method=self.method)
            if threshold is None:
                continue
            entropy1 = self.calcEntropy([row for row in self.data if row[i] < threshold])
            entropy2 = self.calcEntropy([row for row in self.data if row[i] >= threshold])
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
        print(f"Classificando {i+1}/{len(data)}: {data[i]} -> {tree.classify(data[i])} (esperado: {data[i][-1]})")
    return correct / len(data)

def pruning(tree, confidence, z=None, pruning_data=None):
    global NOS_PODADOS
    if z is None:
        z = norm.ppf((1 + confidence) / 2)
    if pruning_data is None:
        pruning_data = tree.data

    for child in tree.children:
        pruning(child, confidence, z, pruning_data)

    if tree.children:
        error_subtree = calcErrorRate(tree, pruning_data, z)
        majority_class = get_majority_class(tree.data)
        error_as_leaf = calcLeafErrorRate(tree, pruning_data, majority_class, z)

        if error_as_leaf <= error_subtree:
            tree.children.clear()
            tree.attribute = None
            tree.threshold = None
            NOS_PODADOS += 1

def get_majority_class(data):
    from collections import Counter
    labels = [row[-1] for row in data]
    return Counter(labels).most_common(1)[0][0]

def calcLeafErrorRate(tree, data, majority_class, z):
    errors = sum(1 for row in data if row[-1] != majority_class)
    f = errors / len(data)
    return f + z * math.sqrt((f * (1 - f)) / len(data))

def calcErrorRate(tree, data, z):
    f = 0
    for i in range(len(data)):
        if int(tree.classify(data[i])) != int(data[i][-1]):
            f += 1
    f = f / len(data)
    return f + z * math.sqrt((f * (1 - f)) / len(data))

def testar_variacoes(data_path="dataset/treino_sinais_vitais_com_label.csv"):
    global NOS_PODADOS
    data = pd.read_csv(data_path).values
    training_percentages = [0.1, 0.3, 0.5, 0.7, 0.9]
    methods = ['median', 'average', 'entropy']
    confidence = 0
    maxDepth = 3

    resultados = {method: {'antes': [], 'depois': []} for method in methods}
    os.makedirs("graficos/c45", exist_ok=True)

    for method in methods:
        for perc in training_percentages:
            treino = data[:int(len(data)*perc)]
            teste = data[int(len(data)*perc):]

            split_index = int(len(treino) * 0.7)
            treino_construcao = treino[:split_index]
            treino_validacao = treino[split_index:]

            tree = c45Node(treino_construcao, maxDepth=maxDepth, method=method)
            tree.build_tree()

            acc_antes = validate(teste, tree)
            NOS_PODADOS = 0
            #print(f"Número de nós antes da poda: {tree.count_nodes()}")
            pruning(tree, confidence=confidence, pruning_data=treino_validacao)
            acc_depois = validate(teste, tree)
            #print(f"Número de nós depois da poda: {tree.count_nodes()}")
            #print(f"[{method.upper()} - {int(perc*100)}% treino] Nós podados: {NOS_PODADOS}")
            #print(f"Acurácias método {method}:")
            #print("Antes:", acc_antes)
            #print("Depois:", acc_depois)

            resultados[method]['antes'].append(acc_antes)
            resultados[method]['depois'].append(acc_depois)

        plt.figure()
        plt.plot(training_percentages, [x * 100 for x in resultados[method]['antes']], label='Antes da poda')
        plt.plot(training_percentages, [x * 100 for x in resultados[method]['depois']], label='Depois da poda')
        plt.title(f"Método: {method}")
        plt.xlabel("Porcentagem de treino")
        plt.ylabel("Acurácia (%)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"graficos/c45/{method}_acuracia.png")
        plt.close()

if __name__ == "__main__":
    testar_variacoes()

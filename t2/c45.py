import math
from collections import Counter
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
import os
import numpy as np

MAX_DEPTH = 0
NOS_PODADOS = 0

class c45Node:
    def __init__(self, data, lvl=0, maxDepth=MAX_DEPTH, entropy=None, method='ganho de informação'):
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
        count = 1
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
        for i in range(1, len(self.data[0])-2):
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
        self.threshold = minEntropyThreshold
        if len(left) == 0 or len(right) == 0:
            self.classification = get_majority_class(self.data)
            return
        self.separateData(att, minEntropyThreshold, entropy1, entropy2)
        for child in self.children:
            child.build_tree()

    def calcThreshold(self, attribute, method='mediana'):
        if method == 'mediana':
            return self.calcMedThreshold(attribute)
        elif method == 'média':
            return self.calcAvgThreshold(attribute)
        elif method == 'ganho de informação':
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
    return correct / len(data)

def pruning(tree, confidence=0.75, z=None):
    global NOS_PODADOS
    if z is None:
        z = norm.ppf((1 + confidence) / 2)
        # print(f"Z-score for confidence {confidence}: {z}")

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

def testar_variacoes(data_path="dataset/treino_sinais_vitais_com_label.csv"):
    global NOS_PODADOS
    data = pd.read_csv(data_path).values
    training_percentages = [0.1, 0.3, 0.5, 0.7, 0.9]
    methods = ['mediana', 'média', 'ganho de informação']
    confidence = 0.69
    maxDepth_values = [1, 3, 6, 9, 12, 15, 18, 21]

    resultados = {
        method: {
            maxDepth: {'antes': [], 'depois': []} for maxDepth in maxDepth_values
        }
        for method in methods
    }

    # Criar diretório para salvar os gráficos
    os.makedirs("graficos/c45", exist_ok=True)

    for method in methods:
        for maxDepth in maxDepth_values:
            for perc in training_percentages:
                treino = data[:int(len(data)*perc)]
                teste = data[int(len(data)*perc):]

                tree = c45Node(treino, maxDepth=maxDepth, method=method)
                tree.build_tree()

                acc_antes = validate(teste, tree)
                NOS_PODADOS = 0
                pruning(tree, confidence=confidence)
                acc_depois = validate(teste, tree)

                resultados[method][maxDepth]['antes'].append(acc_antes)
                resultados[method][maxDepth]['depois'].append(acc_depois)

        # Plotar gráfico para cada método com linhas para cada profundidade
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
        plt.savefig(f"graficos/c45/{method.replace(' ', '_')}_acuracia_comparativa.png")
        plt.close()

    return resultados, training_percentages, maxDepth_values, data


import os

def plot_comparacao_maior_profundidade(resultados, training_percentages, maxDepth_values):
    max_depth = max(maxDepth_values)
    methods = ['mediana', 'média', 'ganho de informação']

    os.makedirs('graficos', exist_ok=True)

    plt.figure(figsize=(10, 6))
    for method in methods:
        acc_antes = [x * 100 for x in resultados[method][max_depth]['antes']]
        acc_depois = [x * 100 for x in resultados[method][max_depth]['depois']]
        plt.plot(training_percentages, acc_antes, linestyle='--', label=f'{method} antes poda')
        plt.plot(training_percentages, acc_depois, linestyle='-', label=f'{method} depois poda')

    plt.title(f'Acurácia x Porcentagem de Treino - Maior Profundidade ({max_depth})')
    plt.xlabel('Porcentagem de treino')
    plt.ylabel('Acurácia (%)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'graficos/c45/acuracia_porcentagem_treino_profundidade_{max_depth}.png')
    plt.close()

def plot_confianca_vs_acuracia(data, maxDepth, method, training_percent=0.5):
    confidences = np.linspace(0.1, 0.99, 10)
    acuracias = []

    treino = data[:int(len(data)*training_percent)]
    teste = data[int(len(data)*training_percent):]

    for conf in confidences:
        global NOS_PODADOS
        NOS_PODADOS = 0
        tree = c45Node(treino, maxDepth=maxDepth, method=method)
        tree.build_tree()
        pruning(tree, confidence=conf)
        acc = validate(teste, tree)
        acuracias.append(acc * 100)
        # print(f"Confiança: {conf:.2f}, Acurácia: {acc * 100:.2f}%, Nós podados: {NOS_PODADOS}")

    os.makedirs('graficos', exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(confidences, acuracias, marker='o')
    plt.title(f'Acurácia x Nível de Confiança (método: {method}, profundidade: {maxDepth})')
    plt.xlabel('Nível de Confiança')
    plt.ylabel('Acurácia (%)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'graficos/c45/acuracia_confianca_metodo_{method}_profundidade_{maxDepth}.png')
    plt.close()

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def salvar_matriz_confusao_csv(matrix, classes, arquivo):
    df_cm = pd.DataFrame(matrix, index=classes, columns=classes)
    df_cm.to_csv(arquivo)

def gerar_matrizes_confusao(data_path="dataset/treino_sinais_vitais_com_label.csv"):
    data = pd.read_csv(data_path).values
    methods = ['mediana', 'média', 'ganho de informação']
    maxDepth = 21  # maior profundidade
    training_percent = 0.5  # maior treino para melhor treino
    
    treino = data[:int(len(data)*training_percent)]
    teste = data[int(len(data)*training_percent):]

    classes = sorted(set(teste[:, -1]))  # classes únicas ordenadas

    os.makedirs("matrizes_confusao", exist_ok=True)

    for method in methods:
        tree = c45Node(treino, maxDepth=maxDepth, method=method)
        tree.build_tree()
        
        # Previsões antes da poda
        preds = []
        for row in teste:
            pred = tree.classify(row)
            preds.append(int(pred))
        preds = np.array(preds)
        
        y_true = teste[:, -1].astype(int)
        
        cm = confusion_matrix(y_true, preds, labels=classes)
        
        arquivo = f"matrizes_confusao/matriz_confusao_{method}_prof{maxDepth}_antes_poda.csv"
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f"Matriz de Confusão - Método: {method}, Profundidade: {maxDepth}")

        # Caminho para salvar o arquivo (crie a pasta se necessário)
        filename_cm = f"graficos/c45/ConfusionMatrix_{method}_Profundidade{maxDepth}.png"
        plt.savefig(filename_cm)
        plt.close()  # Fecha a figura para não mostrar na tela nem consumir memória

if __name__ == "__main__":
    # resultados, training_percentages, maxDepth_values, data = testar_variacoes()
    # plot_comparacao_maior_profundidade(resultados, training_percentages, maxDepth_values)
    data = pd.read_csv("dataset/treino_sinais_vitais_com_label.csv").values
    # Ajuste aqui para o método e profundidade do melhor resultado que você encontrar:
    melhor_metodo = 'ganho de informação'
    melhor_profundidade = 21
    # plot_confianca_vs_acuracia(data, 5, melhor_metodo, training_percent=0.5)
    gerar_matrizes_confusao(data_path="dataset/treino_sinais_vitais_com_label.csv")

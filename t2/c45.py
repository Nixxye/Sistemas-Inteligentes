import math
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
import os

MAX_DEPTH = 5

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
        # self.testData = data[0:len(data) / 2]
        # self.buildData = data[len(data) / 2 : len(data) ]

    def build_tree(self):   
        # Adiciona critério de parada aqui... 
        # 
        if self.lvl > MAX_DEPTH:
            return    
        att = None
        minEntropy = None
        minEntropyThreshold = None
        if len(self.data) == 0:
            return
        for i in range(len(self.data[0])):
            # Ordena os dados para o atributo
            self.data = sorted(self.data, key=lambda x: x[i])
            # Calcula a entropia para cada atributo
            threshold = self.calcThreshold(i, method=self.method)
            if threshold is None:
                # Se não houver threshold, ignora
                continue
            entropy1 = self.calcEntropy([row for row in self.data if row[i] < threshold])
            entropy2 = self.calcEntropy([row for row in self.data if row[i] >= threshold])
            entropy = entropy1 + entropy2
            # Se a entropia for a menor até agora, atualiza
            if minEntropy is None or entropy < minEntropy:
                minEntropy = entropy
                minEntropyThreshold = threshold
                att = i
        # Se não houver atributo, retorna
        if att is None:
            return
        self.attribute = att
        self.threshold = minEntropyThreshold
        # Separa o conjunto de dados em 2
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

    # Mediana  
    def calcMedThreshold(self, attribute):
        # Extrai os valores do atributo (já estão ordenados)
        values = [row[attribute] for row in self.data]
        n = len(values)
        # Calcula a mediana
        if n % 2 == 1:
            return values[n // 2]
        else:
            return (values[n // 2 - 1] + values[n // 2]) / 2
        
    # Média
    def calcAvgThreshold(self, attribute):
        # Extrai os valores do atributo (já estão ordenados, mas não importa para a média)
        values = [row[attribute] for row in self.data]
        # Retorna a média
        return sum(values) / len(values) if values else 0

    # Pela entropia mínima
    def calcEntThreshold(self, attribute):
        best_threshold = None
        best_entropy = float('inf')

        for i in range(len(self.data) - 1):
            class_curr = self.data[i][-1]
            class_next = self.data[i + 1][-1]

            # Só considera pares com classes diferentes
            if class_curr != class_next:
                threshold = (self.data[i][attribute] + self.data[i + 1][attribute]) / 2

                # Separa os dados pelo threshold
                left = [x for x in self.data if x[attribute] < threshold]
                right = [x for x in self.data if x[attribute] >= threshold]

                # Calcula entropia ponderada
                total = len(self.data)
                entropy = (len(left) / total) * self.calcEntropy(left) + (len(right) / total) * self.calcEntropy(right)

                # Atualiza o melhor threshold
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
        # Filtra dados com atributo < threshold
        left_data = [row for row in self.data if row[attribute] < threshold]
        # Filtra dados com atributo >= threshold
        right_data = [row for row in self.data if row[attribute] >= threshold]

        self.children = [
            c45Node(data=left_data, lvl=self.lvl+1, maxDepth=self.maxDepth, entropy=entropy1, method=self.method),
            c45Node(data=right_data, lvl=self.lvl+1, maxDepth=self.maxDepth, entropy=entropy2, method=self.method)
        ]

        
    def classify(self, data):
        if self.children == []:
            # Contar frequência das classes
            class_counts = {}
            for item in self.data:
                label = item[-1]
                class_counts[label] = class_counts.get(label, 0) + 1
            # Retornar a classe mais frequente
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

def pruning(tree, confidence, expectedError, z=None):
    if z is None:
        z = norm.ppf((1 + confidence) / 2)
    averageError = 0
    for child in tree.children:
        if len(child.children) > 0:
            averageError += pruning(child, confidence, expectedError, z) / len(tree.children)
    # Poda
    if averageError > expectedError:
        tree.children.clear()
    
    return calcErrorRate(tree, tree.data, z)

def calcErrorRate(tree, data, z):
    f = 0
    for i in range(len(data)):
        if int(tree.classify(data[i])) != int(data[i][-1]):
            f += 1
    f = f / len(data)
    # Taxa de erro (upper bound)
    return f + z * math.sqrt((f * (1 - f)) / len(data))

def testar_variacoes(data_path="dataset/treino_sinais_vitais_com_label.csv"):
    data = pd.read_csv(data_path).values
    training_percentages = [0.1, 0.3, 0.5, 0.7, 0.9]
    methods = ['median', 'average', 'entropy']
    confidence = 0.2
    expectedError = 0.1
    maxDepth = 3

    resultados = {method: {'antes': [], 'depois': []} for method in methods}

    os.makedirs("graficos/c45", exist_ok=True)

    for method in methods:
        for perc in training_percentages:
            treino = data[:int(len(data)*perc)]
            teste = data[int(len(data)*perc):]

            tree = c45Node(treino, maxDepth=maxDepth, method=method)
            tree.build_tree()
            acc_antes = validate(teste, tree)

            pruning(tree, confidence, expectedError)
            acc_depois = validate(teste, tree)

            resultados[method]['antes'].append(acc_antes)
            resultados[method]['depois'].append(acc_depois)

        # Salvar gráfico
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
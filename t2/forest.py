import math
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
import os
import random
import sys

MAX_DEPTH = 12
NOS_PODADOS = 0

def progress_bar(progress, total, length=50):
    percent = 100 * (progress / total)
    filled = int(length * progress // total)
    bar = '█' * filled + '-' * (length - filled)
    sys.stdout.write(f'\r|{bar}| {percent:.1f}%')
    sys.stdout.flush()

class c45Node:
    def __init__(self, data, lvl=0, maxDepth=MAX_DEPTH, entropy=None, method='entropy', atribRange=None):
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
        if atribRange is None:
            self.range = range(1,len(self.data[0])-2)
        else:
            self.range = atribRange

    def count_nodes(self):
        count = 1  # Conta o nó atual
        for child in self.children:
            count += child.count_nodes()
        return count
    
    def build_tree(self):
        if self.lvl > self.maxDepth or len(self.data) == 0:
            return
        att = None
        minEntropy = None
        minEntropyThreshold = None
        for i in self.range: # Ignora o ID e a classe
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
            c45Node(data=left_data, lvl=self.lvl+1, maxDepth=self.maxDepth, entropy=entropy1, method=self.method, atribRange=self.range),
            c45Node(data=right_data, lvl=self.lvl+1, maxDepth=self.maxDepth, entropy=entropy2, method=self.method, atribRange=self.range)
        ]

    def classify(self, data):
        if self.children == []:
            class_counts = {}
            for item in self.data:
                label = int(item[-1])
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

def testar_variacoes(trainingSize = 1000, treesPop = 1, data_path="dataset/treino_sinais_vitais_com_label.csv"):
    global NOS_PODADOS
    data = pd.read_csv(data_path).values
    confidence = 0
    trees = []
    for i in range(treesPop):
        size = int(trainingSize/treesPop)
        atribRange = None
        tree = c45Node(data[int(i*size):int((i+1)*(size)-1)], maxDepth=MAX_DEPTH, method='entropy', atribRange=atribRange)
        tree.build_tree()
        trees.append(tree)

    corretos = 0
    
    for i in range(trainingSize, 1499):
        bagging = [0,0,0,0]
        for tree in trees:
            bagging[int(tree.classify(data[i])-1)] += 1

        max = 0
        max_index = -1

        for j in range(4):
            if bagging[j] > max:
                max = bagging[j]
                max_index = j

        if max_index+1 == data[i][-1]:
            corretos += 1

        
    # print('resultado de particionamento do treinamento: ', corretos/(1499 - trainingSize))
    return corretos/(1499 - trainingSize)*100
    
def testar_variacoes_random(trainingSize = 1000, treesPop = 1, data_path="dataset/treino_sinais_vitais_com_label.csv"):
    global NOS_PODADOS
    data = pd.read_csv(data_path).values
    confidence = 0
    trees = []
    for i in range(treesPop):
        atribRange = []
        fRange = random.randint(1, len(data[0])-2)
        lRange = random.randint(fRange, len(data[0])-2)
        atribRange = list(range(fRange, lRange))
        if treesPop == 1:
            atribRange = None
        tree = c45Node(data, maxDepth=MAX_DEPTH, method='entropy', atribRange=atribRange)
        tree.build_tree()
        trees.append(tree)

    corretos = 0
    
    for i in range(trainingSize, 1499):
        bagging = [0,0,0,0]
        for tree in trees:
            bagging[int(tree.classify(data[i])-1)] += 1

        max = 0
        max_index = -1

        for j in range(4):
            if bagging[j] > max:
                max = bagging[j]
                max_index = j

        if max_index+1 == data[i][-1]:
            corretos += 1

        
    # print('resultado dos atributos aleátorios: ', corretos/(1499 - trainingSize))
    return corretos/(1499 - trainingSize)*100

def testar_variacoes_random_samples(trainingSize = 1000, treesPop = 1, data_path="dataset/treino_sinais_vitais_com_label.csv"):
    global NOS_PODADOS
    data = pd.read_csv(data_path).values
    confidence = 0
    trees = []
    for i in range(treesPop):
        trainingData = []
        for j in range(trainingSize):
            trainingData.append(data[random.randint(0, trainingSize)])
        atribRange = None
        tree = c45Node(trainingData, maxDepth=MAX_DEPTH, method='entropy', atribRange=atribRange)
        tree.build_tree()
        trees.append(tree)

    corretos = 0
    
    for i in range(trainingSize, 1499):
        bagging = [0,0,0,0]
        for tree in trees:
            bagging[int(tree.classify(data[i])-1)] += 1

        max = 0
        max_index = -1

        for j in range(4):
            if bagging[j] > max:
                max = bagging[j]
                max_index = j

        if max_index+1 == data[i][-1]:
            corretos += 1

        
    # print('resultado dos amostras aleátorias: ', corretos/(1499 - trainingSize))
    return corretos/(1499 - trainingSize)*100

if __name__ == "__main__":
    trainingSize = 500
    treesPop = 1
    iteracoes = 10

    for treesPop in range(1,20):
        y = []
        x = list(range(1, iteracoes+1))
        for i in x:
            y.append(testar_variacoes_random(trainingSize,treesPop))

        plt.plot(x, y, marker='o', color='blue', label='Atributos Aleatórios')



        with open('relatórioAcurácia_AtribRand' + str(treesPop) + '.txt', 'w') as f:
            media = sum(y)/len(y)
            f.write('Média de Acurácia: {:.2f}%\n'.format(media))
            temp = y
            temp.sort()
            f.write('Mediana de Acurácia: {:.2f}%\n'.format(temp[len(y)//2]))
            f.write('Maior Acurácia: {:.2f}%\n' .format(temp[-1]))
            n = len(y)
            desvio = math.sqrt(sum((x - media) ** 2 for x in temp) / (n - 1))
            f.write('Desvio Padrão: {:.2f}%\n'.format(desvio))

        y = []
        for i in x:
            y.append(testar_variacoes_random_samples(trainingSize,treesPop))

        plt.plot(x, y, marker='s', color='red', label='Amostras Aleatórias')

        with open('relatórioAcurácia_SampleRand' + str(treesPop) + '.txt', 'w') as f:
            media = sum(y)/len(y)
            f.write('Média de Acurácia: {:.2f}%\n'.format(media))
            temp = y
            temp.sort()
            f.write('Mediana de Acurácia: {:.2f}%\n'.format(temp[len(y)//2]))
            f.write('Maior Acurácia: {:.2f}%\n' .format(temp[-1]))
            n = len(y)
            desvio = math.sqrt(sum((x - media) ** 2 for x in temp) / (n - 1))
            f.write('Desvio Padrão: {:.2f}%\n'.format(desvio))


        y = []
        for i in x:
            y.append(testar_variacoes(trainingSize,treesPop))

        plt.plot(x, y, marker='^', color='green', label='Particionamento')

        plt.title('Treino {:.2f}% | {:d} Árvores'.format(trainingSize / 1500 * 100, treesPop))
        plt.xlabel('Iteração')
        plt.ylabel('Acurácia')

        plt.grid(True)
        plt.legend()

        plt.savefig('acuráciaXiteração' + str(treesPop) + '.png')

        with open('relatórioAcurácia_Particionamento' + str(treesPop) + '.txt', 'w') as f:
            media = sum(y)/len(y)
            f.write('Média de Acurácia: {:.2f}%\n'.format(media))
            temp = y
            temp.sort()
            f.write('Mediana de Acurácia: {:.2f}%\n'.format(temp[len(y)//2]))
            f.write('Maior Acurácia: {:.2f}%\n' .format(temp[-1]))
            n = len(y)
            desvio = math.sqrt(sum((x - media) ** 2 for x in temp) / (n - 1))
            f.write('Desvio Padrão: {:.2f}%\n'.format(desvio))
        plt.close()
        progress_bar(treesPop, 20)

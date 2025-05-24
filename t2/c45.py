import math
import pandas as pd

MAX_DEPTH = 5

class c45Node:
    def __init__(self, data, lvl=0):
        
        self.lvl = lvl
        self.children = []
        self.attribute = None
        self.threshold = None
        self.data = data
        # self.testData = data[0:len(data) / 2]
        # self.buildData = data[len(data) / 2 : len(data) ]

    def build_tree(self):   
        # Adiciona critério de parada aqui... 
        # 
        if self.lvl > MAX_DEPTH:
            return    
        att = 0
        minEntropy = None
        minEntropyThreshold = None
        if len(self.data) == 0:
            return
        for i in range(len(self.data[0])):
            # Ordena os dados para o atributo
            self.data = sorted(self.data, key=lambda x: x[i])
            # Calcula a entropia para cada atributo
            entropy, threshold = self.calcEntropy(i)
            # Se a entropia for a menor até agora, atualiza
            if minEntropy is None or entropy < minEntropy:
                minEntropy = entropy
                minEntropyThreshold = threshold
                att = i

        self.attribute = att
        self.threshold = minEntropyThreshold
        # Separa o conjunto de dados em 2
        self.separateData(att, minEntropyThreshold)
        for child in self.children:
            child.build_tree()
        
        
    # Por enquanto, divide os dados no meio e calcula a entropia para cada metade
    def calcEntropy(self, attribute):
        ent1, ent2 = 0, 0
        frequency = [0, 0, 0, 0]
        for i in range(int(len(self.data) / 2)):
            frequency[int(self.data[i][-1])-1] += 1
        for x in frequency:
            if x > 0:
                f = 2 * x / len(self.data)
                ent1 -= f * math.log2(f)
        frequency = [0, 0, 0, 0]

        for i in range(int(len(self.data) / 2), int(len(self.data))):
            frequency[int(self.data[i][-1])-1] += 1
        for x in frequency:
            if x > 0:
                f = 2 * x / len(self.data)
                ent2 -= f * math.log2(f)
                
        return ent1 + ent2, self.data[int(len(self.data) / 2)][attribute]

    def separateData(self, attribute, threshold):
        self.data = sorted(self.data, key=lambda x: x[attribute])
        self.children = [c45Node(self.data[0: len(self.data) // 2], self.lvl+1),
                          c45Node(self.data[len(self.data) // 2 : len(self.data)], self.lvl+1)]
        
    def classify(self, data):
        if self.children == []:
            #print("Classificado como: ", self.data[0][-1])
            return self.data[0][-1]
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

if __name__ == "__main__":
    data = pd.read_csv("dataset/treino_sinais_vitais_com_label.csv")
    trainingPercentage = 0.2
    arvre = c45Node(data.values[0: int(len(data.values) * trainingPercentage)])
    arvre.build_tree()
    print("Acurácia: ", validate(data.values[int(len(data.values) * trainingPercentage):], arvre))
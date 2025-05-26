import random
import numpy as np
import math
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score

# Defina os valores fixos que você deseja usar
LEARNING_RATE = 0.3
HIDDEN_LAYERS_SIZES = [1, 3, 5]
NEURONS_PER_LAYERS = [1, 3, 5]
TRAINING_PERCENTAGES = [0.3]
EPOCHS = 300

def testar_neural_network(data_path="dataset/treino_sinais_vitais_com_label.csv"):
    data = pd.read_csv(data_path).values
    training_percentages = TRAINING_PERCENTAGES

    os.makedirs("graficos/neural", exist_ok=True)

    encoder = OneHotEncoder(sparse_output=False)
    all_labels = data[:, -1].reshape(-1, 1)
    encoder.fit(all_labels)

    resultados = []  # tuplas (label, perc, acc_before, acc_after)

    for perc in training_percentages:
        treino = data[:int(len(data) * perc)]
        teste = data[int(len(data) * perc):]

        X_train, y_train = treino[:, :-1], treino[:, -1].reshape(-1, 1)
        X_test, y_test = teste[:, :-1], teste[:, -1].reshape(-1, 1)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        y_train_encoded = encoder.transform(y_train)
        y_test_int = y_test.astype(int).flatten()

        for hidden_layers in HIDDEN_LAYERS_SIZES:
            for neurons_per_layer in NEURONS_PER_LAYERS:
                print(f"\nTreino {int(perc*100)}% | {hidden_layers} camadas, {neurons_per_layer} neurônios")

                nn = NeuralNetwork(
                    inputSize=X_train.shape[1],
                    numClasses=y_train_encoded.shape[1],
                    hiddenLayersSize=hidden_layers,
                    neuronsPerLayer=neurons_per_layer
                )

                y_pred_before = [nn.predictClass(x) for x in X_test]
                acc_before = accuracy_score(y_test_int, y_pred_before)

                for epoch in range(EPOCHS):
                    for x, y in zip(X_train, y_train_encoded):
                        nn.train(x, y)

                y_pred_after = [nn.predictClass(x) for x in X_test]
                acc_after = accuracy_score(y_test_int, y_pred_after)

                print(f"Acurácia antes: {acc_before*100:.2f}%, depois: {acc_after*100:.2f}%")

                label = f"{hidden_layers}x{neurons_per_layer}"
                resultados.append((label, f"{int(perc*100)}%", acc_before*100, acc_after*100))

    # Criar um gráfico para cada % de treino, com barras duplas para cada configuração
    for perc in training_percentages:
        labels = []
        acc_before_vals = []
        acc_after_vals = []
        for (label, p, acc_b, acc_a) in resultados:
            if p == f"{int(perc*100)}%":
                labels.append(label)
                acc_before_vals.append(acc_b)
                acc_after_vals.append(acc_a)

        x = np.arange(len(labels))
        width = 0.35

        plt.figure(figsize=(12, 6))
        plt.bar(x - width/2, acc_before_vals, width=width, label="Antes do treino", color="gray")
        plt.bar(x + width/2, acc_after_vals, width=width, label="Depois do treino", color="green")

        plt.xticks(x, labels, rotation=45)
        plt.xlabel("Configuração (camadas x neurônios)")
        plt.ylabel("Acurácia (%)")
        plt.title(f"Acurácia para {int(perc*100)}% de treino")
        plt.ylim(0, 100)
        plt.grid(axis="y")
        plt.legend()
        plt.tight_layout()

        plt.savefig(f"graficos/neural/acuracia_{int(perc*100)}pct.png")
        plt.close()


class Perceptron:
    def __init__(self, inputSize):
        self.weights = np.random.uniform(-1, 1, inputSize)
        self.bias = random.uniform(-1, 1)
        self.learningRate = LEARNING_RATE
        self.inputSize = inputSize
        self.lastOutput = None
        self.lastInputs = None
        self.delta = None
    
    def predict(self, inputs):
        self.lastInputs = np.array(inputs)
        if len(inputs) != len(self.weights):
            raise ValueError("Input size must match weights size.")
        
        weightedSum = np.dot(inputs, self.weights) + self.bias
        self.lastOutput = 1 / (1 + math.exp(-weightedSum))  # Sigmoid activation function
        return self.lastOutput
    
    def updateWeights(self, error):
        self.delta = self.lastOutput * (1 - self.lastOutput) * error
        for i in range(self.inputSize):
            old = self.weights[i]
            self.weights[i] += self.learningRate * self.delta * self.lastInputs[i]
            # if old != self.weights[i]:
            #     print(f"Peso {i} alterado: {old:.4f} -> {self.weights[i]:.4f}")
        self.bias += self.learningRate * self.delta

class OutputLayer:
    def __init__ (self, inputSize, numClasses):
        self.weights = np.random.uniform(-1, 1, (numClasses, inputSize)) # Vetor de pesos para cada classe
        self.bias = np.random.uniform(-1, 1, numClasses)
        self.learningRate = LEARNING_RATE
        self.numClasses = numClasses
        self.lastOutputs = None
        self.lastInputs = None
        self.deltas = []

    def predict(self, inputs):
        if len(inputs) != self.weights.shape[1]:  # <- shape[1] é inputSize
            raise ValueError("Input size must match weights size.")
    
        self.lastInputs = np.array(inputs)
        self.lastOutputs = []
        weightedSum = 0

        for i in range(len(self.weights)):
            weightedSum = np.dot(self.weights[i], inputs) + self.bias[i]
            output = 1 / (1 + math.exp(-weightedSum))
            self.lastOutputs.append(output)

        return self.lastOutputs.index(max(self.lastOutputs)) + 1

    def updateWeights(self, target):
        self.deltas = [self.lastOutputs[i] * (1 - self.lastOutputs[i] ) * (target[i] - self.lastOutputs[i])for i in range(self.numClasses)]
        for i in range(self.numClasses):
            self.weights[i] += self.learningRate * self.deltas[i] * self.lastInputs

class NeuralNetwork:
    def __init__(self, inputSize, numClasses, hiddenLayersSize=1, neuronsPerLayer=1):
        self.inputSize = inputSize
        self.numClasses = numClasses
        self.outputLayer = OutputLayer(neuronsPerLayer, numClasses)
        self.hiddenLayers = []
        self.createLayers(hiddenLayersSize, neuronsPerLayer)
    
    def createLayers(self, hiddenLayersSize, neuronsPerLayer):
        n = self.inputSize
        for _ in range(hiddenLayersSize):
            layer = []
            for _ in range(neuronsPerLayer):
                layer.append(Perceptron(n))
            n = neuronsPerLayer
            self.hiddenLayers.append(layer)

    def feedForward(self, inputs):
        for layer in self.hiddenLayers:
            outputs = []
            for neuron in layer:
                outputs.append(neuron.predict(inputs))
            inputs = outputs
        return self.outputLayer.predict(inputs)
    
    def backpropagate(self, target):
        # Update output layer
        self.outputLayer.updateWeights(target)

        nextDeltas = self.outputLayer.deltas
        nextWeights = self.outputLayer.weights

        for layer in reversed(self.hiddenLayers):
            currentDeltas = []

            for j, neuron in enumerate(layer):
                error = 0

                for k in range(len(nextDeltas)):
                    error += nextDeltas[k] * nextWeights[k][j]

                neuron.updateWeights(error)

                currentDeltas.append(neuron.delta)

            nextDeltas = currentDeltas
            nextWeights = [neuron.weights for neuron in layer]

    def train(self, inputs, target):
        inputs = np.array(inputs)
        target = np.array(target)
        if len(inputs) != self.inputSize:
            raise ValueError("Input size must match the network input size.")
        
        self.feedForward(inputs)
        self.backpropagate(target)

    def predictClass(self, inputs):
        return self.feedForward(inputs)
    
    def getAllWeights(self):
        weights = {"output": self.outputLayer.weights.tolist()}
        for i, layer in enumerate(self.hiddenLayers):
            weights[f"hidden_{i}"] = [neuron.weights.tolist() for neuron in layer]
        return weights

if __name__ == "__main__":
    testar_neural_network("dataset/treino_sinais_vitais_com_label.csv")
import random
import numpy as np
import math
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score



LEARNING_RATES = [0.1]
HIDDEN_LAYERS_SIZES = [5]
NEURONS_PER_LAYERS = [5]
TRAINING_PERCENTAGES = [0.3]
EPOCHS = 300
METHODS = ['sigmoid', 'tanh', 'relu']


def testar_neural_network(data_path="dataset/treino_sinais_vitais_com_label.csv"):
    data = pd.read_csv(data_path).values
    os.makedirs("graficos/neural", exist_ok=True)

    encoder = OneHotEncoder(sparse_output=False)
    encoder.fit(data[:, -1].reshape(-1, 1))

    resultados = []

    for perc in TRAINING_PERCENTAGES:
        treino = data[:int(len(data) * perc)]
        teste = data[int(len(data) * perc):]

        X_train, y_train = treino[:, :-1], treino[:, -1].reshape(-1, 1)
        X_test, y_test = teste[:, :-1], teste[:, -1].reshape(-1, 1)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        y_train_encoded = encoder.transform(y_train)
        y_test_int = y_test.astype(int).flatten()

        # Criar rede base com sigmoid e armazenar pesos/bias iniciais
        base_nn = NeuralNetwork(
            inputSize=X_train.shape[1],
            numClasses=y_train_encoded.shape[1],
            hiddenLayersSize=HIDDEN_LAYERS_SIZES[0],
            neuronsPerLayer=NEURONS_PER_LAYERS[0],
            learningRate=LEARNING_RATES[0],
            method='sigmoid'
        )
        initial_weights = base_nn.getAllWeights()

        for method in METHODS:
            for hidden_layers in HIDDEN_LAYERS_SIZES:
                for neurons_per_layer in NEURONS_PER_LAYERS:
                    for learning_rate in LEARNING_RATES:
                        print(f"\nMétodo: {method} | Treino {int(perc*100)}% | {hidden_layers} camadas, {neurons_per_layer} neurônios, LR={learning_rate}")

                        nn = NeuralNetwork(
                            inputSize=X_train.shape[1],
                            numClasses=y_train_encoded.shape[1],
                            hiddenLayersSize=hidden_layers,
                            neuronsPerLayer=neurons_per_layer,
                            learningRate=learning_rate,
                            method=method
                        )
                        # Aplicar pesos/bias iniciais para garantir igualdade
                        nn.setAllWeights(initial_weights)


                        y_pred_before = [nn.predictClass(x) for x in X_test]
                        acc_before = accuracy_score(y_test_int, y_pred_before)

                        for _ in range(EPOCHS):
                            for x, y in zip(X_train, y_train_encoded):
                                nn.train(x, y)

                        y_pred_after = [nn.predictClass(x) for x in X_test]
                        acc_after = accuracy_score(y_test_int, y_pred_after)

                        print(f"Acurácia antes: {acc_before*100:.2f}%, depois: {acc_after*100:.2f}%")

                        resultados.append((method, acc_before*100, acc_after*100))

    # Gráfico comparativo
    labels, accs_before, accs_after = zip(*resultados)
    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(8, 6))
    plt.bar(x - width/2, accs_before, width=width, label="Antes", color="gray")
    plt.bar(x + width/2, accs_after, width=width, label="Depois", color="green")
    plt.xticks(x, labels)
    plt.ylabel("Acurácia (%)")
    plt.title("Comparação de funções de ativação")
    plt.ylim(0, 100)
    plt.grid(axis="y")
    plt.legend()
    plt.tight_layout()
    plt.savefig("graficos/neural/comparativo_metodos_ativacao.png")
    plt.close()




class Perceptron:
    def __init__(self, inputSize, learningRate, method='sigmoid'):
        self.weights = np.random.uniform(-1, 1, inputSize)
        self.bias = random.uniform(-1, 1)
        self.learningRate = learningRate
        self.inputSize = inputSize
        self.lastOutput = None
        self.lastInputs = None
        self.delta = None
        self.method = method
    
    def activationFunction(self, x):
        if self.method == 'sigmoid':
            return 1 / (1 + math.exp(-x))
        elif self.method == 'relu':
            return max(0, x)
        elif self.method == 'tanh':
            return math.tanh(x)
        else:
            raise ValueError("Método de ativação desconhecido. Use 'sigmoid', 'relu' ou 'tanh'.")

    def predict(self, inputs):
        self.lastInputs = np.array(inputs)
        if len(inputs) != len(self.weights):
            raise ValueError("Input size must match weights size.")
        
        weightedSum = np.dot(inputs, self.weights) + self.bias
        self.lastOutput = self.activationFunction(weightedSum)
        return self.lastOutput
    
    def calculateDelta(self, error):
        if self.method == 'sigmoid':
            return self.lastOutput * (1 - self.lastOutput) * error
        elif self.method == 'relu':
            return (self.lastOutput > 0).astype(float) * error
        elif self.method == 'tanh':
            return (1 - math.tanh(self.lastOutput)**2) * error
        else:
            raise ValueError("Método de ativação desconhecido. Use 'sigmoid', 'relu' ou 'tanh'.")

    def updateWeights(self, error):
        self.delta = self.lastOutput * (1 - self.lastOutput) * error
        for i in range(self.inputSize):
            old = self.weights[i]
            self.weights[i] += self.learningRate * self.delta * self.lastInputs[i]
            # if old != self.weights[i]:
            #     print(f"Peso {i} alterado: {old:.4f} -> {self.weights[i]:.4f}")
        self.bias += self.learningRate * self.delta

class OutputLayer:
    def __init__ (self, inputSize, numClasses, learningRate):
        self.weights = np.random.uniform(-1, 1, (numClasses, inputSize)) # Vetor de pesos para cada classe
        self.bias = np.random.uniform(-1, 1, numClasses)
        self.learningRate = learningRate
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
    def __init__(self, inputSize, numClasses, hiddenLayersSize=1, neuronsPerLayer=1, learningRate=0.01, method='sigmoid'):
        self.inputSize = inputSize
        self.numClasses = numClasses
        self.learningRate = learningRate
        self.outputLayer = OutputLayer(neuronsPerLayer, numClasses, self.learningRate)
        self.hiddenLayers = []
        self.method = method
        self.createLayers(hiddenLayersSize, neuronsPerLayer)
    
    def createLayers(self, hiddenLayersSize, neuronsPerLayer):
        n = self.inputSize
        for _ in range(hiddenLayersSize):
            layer = []
            for _ in range(neuronsPerLayer):
                layer.append(Perceptron(n, self.learningRate, self.method))
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
    #Para testes
    def setAllWeights(self, all_weights):
        # Output layer
        self.outputLayer.weights = np.array(all_weights["output"])
        
        # Hidden layers
        for i, layer in enumerate(self.hiddenLayers):
            for j, neuron in enumerate(layer):
                neuron.weights = np.array(all_weights[f"hidden_{i}"][j])

if __name__ == "__main__":
    testar_neural_network("dataset/treino_sinais_vitais_com_label.csv")
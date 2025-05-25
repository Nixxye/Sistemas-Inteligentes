import math
import random
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score


inputSize = 7
METHOD = 'sigmoid'

class NeuralNetwork:
    def __init__(self, inputSize=inputSize, numLayers=3, numNeurons=3):
        self.inputSize = inputSize

        self.layers = []
        self.numLayers = numLayers
        self.numNeurons = numNeurons
        self.layerOutputs = []
        self.outputLayer = OutputLayer(numNeurons)

    def createLayers(self):
        for i in range(self.numLayers):
            layer = []
            for j in range(self.numNeurons):
                neuron = Perceptron(self.inputSize)
                neuron.weights = [random.uniform(-1, 1) for _ in range(self.inputSize)]
                layer.append(neuron)
            self.layers.append(layer)

    def feedForward(self, inputs):
        if len(inputs) != self.inputSize:
            raise ValueError(f"Tamanho do vetor de entrada ({len(inputs)}) "
                             f"incompatível com inputSize ({self.inputSize})")

        self.layerOutputs = []
        for layer in self.layers:
            layerOutputs = []
            for neuron in layer:
                netInput = neuron.calculateNetInput(inputs)
                output = neuron.activationFunction(netInput)
                layerOutputs.append(output)
            self.layerOutputs.append(layerOutputs)
            inputs = layerOutputs

        self.finalOutputs = self.outputLayer.getProbabilites(layerOutputs)
        return self.finalOutputs

    def train(self, inputs, target):
        # Forward pass
        outputs = self.feedForward(inputs)

        # Compute deltas for output layer
        outputDeltas = self.outputLayer.computeOutputDeltas(outputs, target)

        # Update output layer weights
        self.outputLayer.updateWeights(self.layerOutputs[-1], outputDeltas)

        # Backpropagate through hidden layers
        nextDeltas = outputDeltas
        nextLayer = self.outputLayer
        for l in reversed(range(len(self.layers))):
            currentLayer = self.layers[l]
            currentOutputs = self.layerOutputs[l]
            currentDeltas = []
            for i, neuron in enumerate(currentLayer):
                error = 0.0
                for j in range(len(nextDeltas)):
                    if isinstance(nextLayer, OutputLayer):
                        error += nextLayer.weights[j][i] * nextDeltas[j]
                    else:
                        error += nextLayer[j].weights[i] * nextDeltas[j]
                delta = error * currentOutputs[i] * (1 - currentOutputs[i])
                neuron.updateWeights(inputs if l == 0 else self.layerOutputs[l - 1], delta)
                currentDeltas.append(delta)
            nextDeltas = currentDeltas
            nextLayer = currentLayer

    def predictClass(self, inputsVector):
        inputList = list(inputsVector)
        if len(inputsVector) != self.inputSize:
            raise ValueError(f"NeuralNetwork: Tamanho do vetor de entrada ({len(inputsVector)}) "
                             f"incompatível com inputSize da rede ({self.inputSize})")

        for hiddenLayer in self.layers:
            nextLayerInputs = []
            for neuron in hiddenLayer:
                netInput = neuron.calculateNetInput(inputList)
                output = neuron.activationFunction(netInput)
                nextLayerInputs.append(output)
            inputList = nextLayerInputs

        if self.outputLayer:
            return self.outputLayer.getPredictedClass(inputList)
        else:
            raise RuntimeError("A camada de saída (OutputLayer) não foi inicializada corretamente.")

class OutputLayer:
    def __init__(self, inputSize, numClasses=4):
        self.inputSize = inputSize
        self.numClasses = numClasses
        self.weights = [[random.uniform(-1, 1) for _ in range(inputSize)] for _ in range(numClasses)]
        self.biases = [0.0] * numClasses
        self.learningRate = 0.01

    def softmax(self, z):
        max_z = max(z)
        exp_values = [math.exp(x - max_z) for x in z]
        total = sum(exp_values)
        return [val / total for val in exp_values]

    def getProbabilites(self, inputs):
        z = []
        for i in range(self.numClasses):
            weighted_sum = sum(inputs[j] * self.weights[i][j] for j in range(self.inputSize)) + self.biases[i]
            z.append(weighted_sum)
        return self.softmax(z)

    def getPredictedClass(self, inputs):
        probs = self.getProbabilites(inputs)
        return probs.index(max(probs)) + 1

    def computeOutputDeltas(self, predicted, target):
        return [predicted[i] - target[i] for i in range(self.numClasses)]

    def updateWeights(self, inputs, deltas):
        for i in range(self.numClasses):
            for j in range(self.inputSize):
                self.weights[i][j] -= self.learningRate * deltas[i] * inputs[j]
            self.biases[i] -= self.learningRate * deltas[i]

class Perceptron:
    def __init__(self, inputSize, method=METHOD):
        self.weights = [random.uniform(-1, 1) for _ in range(inputSize)]
        self.bias = 0.0
        self.learningRate = 0.01
        self.inputSize = inputSize
        self.method = method
        self.lastOutput = 0

    def calculateNetInput(self, inputsVector):
        if len(inputsVector) != self.inputSize:
            raise ValueError(f"Tamanho do vetor de entrada ({len(inputsVector)}) "
                             f"incompatível com inputSize ({self.inputSize})")

        netInput = sum(self.weights[i] * inputsVector[i] for i in range(self.inputSize)) + self.bias
        return netInput

    def updateWeights(self, inputs, delta):
        for i in range(self.inputSize):
            self.weights[i] -= self.learningRate * delta * inputs[i]
        self.bias -= self.learningRate * delta

    def activationFunction(self, x):
        if self.method == 'sigmoid':
            self.lastOutput = self.sigmoid(x)
        elif self.method == 'linear':
            self.lastOutput = x
        else:
            raise ValueError("Método de ativação desconhecido")
        return self.lastOutput

    def sigmoid(self, x):
        try:
            return 1 / (1 + math.exp(-x))
        except OverflowError:
            return 0.0 if x < 0 else 1.0




def testar_neural_network(data_path="dataset/treino_sinais_vitais_com_label.csv"):
    data = pd.read_csv(data_path).values
    training_percentages = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    resultados = []

    os.makedirs("graficos/neural", exist_ok=True)

    encoder = OneHotEncoder(sparse_output=False)
    all_labels = data[:, -1].reshape(-1, 1)
    encoder.fit(all_labels)

    for perc in training_percentages:
        treino = data[:int(len(data)*perc)]
        teste = data[int(len(data)*perc):]

        X_train, y_train = treino[:, :-1], treino[:, -1].reshape(-1, 1)
        X_test, y_test = teste[:, :-1], teste[:, -1].reshape(-1, 1)

        y_train_encoded = encoder.transform(y_train)
        y_test_int = y_test.astype(int).flatten()

        # Criar e treinar a rede
        nn = NeuralNetwork(inputSize=X_train.shape[1], numLayers=1, numNeurons=6)
        nn.createLayers()

        for epoch in range(100):  # Pode ajustar o número de épocas
            for x, y in zip(X_train, y_train_encoded):
                nn.train(x, y)

        # Testar
        y_pred = [nn.predictClass(x) for x in X_test]
        acc = accuracy_score(y_test_int, y_pred)
        resultados.append(acc)

    # Salvar gráfico
    plt.figure()
    plt.plot(training_percentages, [x * 100 for x in resultados], marker='o', label='Rede Neural')
    plt.title("Acurácia da Rede Neural")
    plt.xlabel("Porcentagem de treino")
    plt.ylabel("Acurácia (%)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("graficos/neural/rede_neural_acuracia.png")
    plt.close()

if __name__ == "__main__":
    # Testar a rede neural
    testar_neural_network()
import math
import random
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


inputSize = 7
METHOD = 'sigmoid'
LEARNING_RATE = 0.3

class NeuralNetwork:
    def __init__(self, inputSize=inputSize, numLayers=3, numNeurons=3):
        self.inputSize = inputSize

        self.layers = []
        self.numLayers = numLayers
        self.numNeurons = numNeurons
        self.layerOutputs = []
        self.outputLayer = OutputLayer(numNeurons)

    def printWeights(self, titulo="Pesos da rede"):
        print(f"\n=== {titulo} ===")
        for l_idx, layer in enumerate(self.layers):
            print(f"Camada {l_idx}:")
            for n_idx, neuron in enumerate(layer):
                print(f"  Neurônio {n_idx}: Pesos = {neuron.weights}, Bias = {neuron.bias}")
        print("Camada de saída:")
        for c_idx, weights in enumerate(self.outputLayer.weights):
            print(f"  Classe {c_idx}: Pesos = {weights}, Bias = {self.outputLayer.biases[c_idx]}")

    def createLayers(self):
        previousSize = self.inputSize
        for i in range(self.numLayers):
            layer = []
            for j in range(self.numNeurons):
                neuron = Perceptron(previousSize)
                neuron.weights = [random.uniform(-0.1, 0.1) for _ in range(previousSize)]
                layer.append(neuron)
            self.layers.append(layer)
            previousSize = self.numNeurons  # Atualiza para a próxima camada
        self.outputLayer = OutputLayer(previousSize)  # Corrige tamanho da camada de saída


    def feedForward(self, inputs):
        if len(inputs) != self.inputSize:
            raise ValueError(f"Tamanho do vetor de entrada ({len(inputs)}) "
                             f"incompatível com inputSize ({self.inputSize})")

        self.layersOutputs = []
        i = 0
        for layer in self.layers:
            layerOutputs = []
            for neuron in layer:
                output = neuron.activationFunction(inputs)
                layerOutputs.append(output)
            self.layersOutputs.append(layerOutputs)
            inputs = layerOutputs
            i += 1

        self.finalOutputs = self.outputLayer.getProbabilites(inputs)
        return self.finalOutputs

    def train(self, inputs, target):
        # Forward pass
        outputs = self.feedForward(inputs)

        # Update output layer weights
        self.outputLayer.updateWeights(target)

        # Backpropagate through hidden layers
        self.deltas = self.outputLayer.deltas
        nextLayer = self.outputLayer
        for i in reversed(range(len(self.layers))):
            currentLayer = self.layers[i]
            currentOutputs = self.layersOutputs[i]
            currentDeltas = []
            for j, neuron in enumerate(currentLayer):
                # Calcular erro: soma dos deltas da próxima camada * pesos
                error = 0.0
                if isinstance(nextLayer, OutputLayer):
                    for k in range(len(nextLayer.deltas)):
                        error += nextLayer.weights[j][k] * nextLayer.deltas[k]
                else:
                    for k, nextNeuron in enumerate(nextLayer):
                        error += nextNeuron.weights[j] * nextNeuron.delta
                
                # Atualizar pesos e guardar delta
                neuron.updateWeights(error)
                currentDeltas.append(neuron.delta)
            
            # Atualizar nextLayer e deltas para próxima iteração
            nextLayer = currentLayer
            self.deltas = currentDeltas



    def predictClass(self, inputsVector):
        inputList = list(inputsVector)
        if len(inputsVector) != self.inputSize:
            raise ValueError(f"NeuralNetwork: Tamanho do vetor de entrada ({len(inputsVector)}) "
                             f"incompatível com inputSize da rede ({self.inputSize})")
        nextinputList = []
        for hiddenLayer in self.layers:
            for neuron in hiddenLayer:
                output = neuron.activationFunction(inputList)
                nextinputList.append(output)
            inputList = nextinputList
            nextinputList = []

        if self.outputLayer:
            return self.outputLayer.getPredictedClass(inputList)
        else:
            raise RuntimeError("A camada de saída (OutputLayer) não foi inicializada corretamente.")

class OutputLayer:
    def __init__(self, inputSize, numClasses=4):
        self.inputSize = inputSize
        self.numClasses = numClasses
        self.weights = [[random.uniform(-0.1, 0.1) for _ in range(inputSize)] for _ in range(numClasses)]
        self.biases = [0.0] * numClasses
        self.learningRate = LEARNING_RATE
        self.lastInput = []
        self.deltas = []
        self.lastOutputs = []

    def softmax(self, z):
        max_z = max(z)
        exp_values = [math.exp(x - max_z) for x in z]
        total = sum(exp_values)
        return [val / total for val in exp_values]

    def getProbabilites(self, inputs):
        self.lastInput = inputs
        z = []
        for i in range(self.numClasses):
            weighted_sum = sum(inputs[j] * self.weights[i][j] for j in range(self.inputSize)) + self.biases[i]
            z.append(weighted_sum)
        self.lastOutputs = self.softmax(z)

        return self.lastOutputs

    def getPredictedClass(self, inputs):
        probs = self.getProbabilites(inputs)
        return probs.index(max(probs)) + 1

    def computeOutputDeltas(self, targets):
        self.deltas = [self.lastOutputs[i] - targets[i] for i in range(self.numClasses)]

    def updateWeights(self, targets):
        self.computeOutputDeltas(targets)
        for class_idx in range(self.numClasses):
            for i in range(self.inputSize):
                old = self.weights[class_idx][i]
                self.weights[class_idx][i] -= self.learningRate * self.deltas[class_idx] * self.lastInput[i]
                # if old != self.weights[class_idx][i]:
                #     print(f"Peso [{class_idx}][{i}] alterado: {old:.4f} -> {self.weights[class_idx][i]:.4f}")
            self.biases[class_idx] -= self.learningRate * self.deltas[class_idx]




class Perceptron:
    def __init__(self, inputSize, method=METHOD):
        self.weights = [random.uniform(-0.1, 0.1) for _ in range(inputSize)]
        self.bias = random.uniform(-0.1, 0.1)
        self.learningRate = LEARNING_RATE
        self.inputSize = inputSize
        self.method = method
        self.lastOutput = 0
        self.lastInputs = []
        self.delta = 0
        self.netInput = 0

    def calculateNetInput(self, inputsVector):
        self.lastInputs = inputsVector
        if len(inputsVector) != self.inputSize:
            raise ValueError(f"Tamanho do vetor de entrada ({len(inputsVector)}) "
                             f"incompatível com inputSize ({self.inputSize})")

        self.netInput = sum(self.weights[i] * inputsVector[i] for i in range(self.inputSize)) + self.bias


    def computeDelta(self, error): # VERIFICAR DERIVADAS
        if self.method == 'sigmoid':
            # Derivada da sigmoid: output * (1 - output)
            self.delta = error * self.lastOutput * (1 - self.lastOutput)
        elif self.method == 'linear':
            # Derivada da função linear é 1
            self.delta = error            

    def updateWeights(self, error):
        self.computeDelta(error)
        for i in range(self.inputSize):
            old = self.weights[i]
            self.weights[i] -= self.learningRate * self.delta * self.lastInputs[i]
            # if old != self.weights[i]:
            #     print(f"Peso {i} alterado: {old:.4f} -> {self.weights[i]:.4f}")
        self.bias -= self.learningRate * self.delta


    def activationFunction(self, inputsVector):
        self.calculateNetInput(inputsVector)
        x = self.netInput
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
    training_percentages = [0.1, 0.5, 0.9]
    
    resultados = []

    os.makedirs("graficos/neural", exist_ok=True)

    encoder = OneHotEncoder(sparse_output=False)
    all_labels = data[:, -1].reshape(-1, 1)
    encoder.fit(all_labels) # 2 -> [0, 1, 0, 0] (transforma as classes em vetores)

    for perc in training_percentages:
        treino = data[:int(len(data)*perc)]
        teste = data[int(len(data)*perc):]
        # Separa as entradas e as classificações
        X_train, y_train = treino[:, :-1], treino[:, -1].reshape(-1, 1)
        X_test, y_test = teste[:, :-1], teste[:, -1].reshape(-1, 1)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        # Converte as classificações para vetores
        y_train_encoded = encoder.transform(y_train)
        y_test_int = y_test.astype(int).flatten()

        # Criar e treinar a rede
        nn = NeuralNetwork(inputSize=X_train.shape[1], numLayers=3, numNeurons=2)
        nn.createLayers()

        # nn.printWeights("Pesos iniciais")

        for epoch in range(100):
            for x, y in zip(X_train, y_train_encoded):
                nn.train(x, y)
        # nn.printWeights("Pesos finais")
        # Testar
        y_pred = [nn.predictClass(x) for x in X_test]
        acc = accuracy_score(y_test_int, y_pred)
        resultados.append(acc)
        print(f"Acurácia para {int(perc * 100)}% de treino: {acc * 100:.2f}%")

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
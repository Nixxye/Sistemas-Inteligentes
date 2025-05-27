import random
import numpy as np
import math
import pandas as pd
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # Importação necessária para plots 3D
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score

# Valores de exemplo para gerar um gráfico 3D mais informativo
LEARNING_RATES = [0.1]
HIDDEN_LAYERS_SIZES = [1, 2, 3]  # Mais de um valor para o eixo Y do gráfico 3D
NEURONS_PER_LAYERS = [1, 3, 5] # Mais de um valor para o eixo X do gráfico 3D
TRAINING_PERCENTAGES = [0.5]
EPOCHS = 100 # Reduzido para fins de demonstração rápida. Pode aumentar conforme necessário.
METHODS = ['sigmoid']

def testar_neural_network(data_path="dataset/treino_sinais_vitais_com_label.csv"):
    data = pd.read_csv(data_path).values
    os.makedirs("graficos/neural", exist_ok=True)

    encoder = OneHotEncoder(sparse_output=False)
    encoder.fit(data[:, -1].reshape(-1, 1))

    plot_data = []

    for perc in TRAINING_PERCENTAGES:
        treino = data[:int(len(data) * perc)]
        teste = data[int(len(data) * perc):]

        X_train, y_train_labels = treino[:, :-1], treino[:, -1].reshape(-1, 1)
        X_test, y_test_labels = teste[:, :-1], teste[:, -1].reshape(-1, 1)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        y_train_encoded = encoder.transform(y_train_labels)
        y_test_int = y_test_labels.astype(float).astype(int).flatten()

        for method in METHODS:
            for hidden_layers_config in HIDDEN_LAYERS_SIZES: # Renomeado para clareza
                for neurons_per_layer_config in NEURONS_PER_LAYERS: # Renomeado para clareza
                    for learning_rate in LEARNING_RATES:
                        print(f"\nMétodo: {method} | Treino {int(perc*100)}% | {hidden_layers_config} camadas, {neurons_per_layer_config} neurônios, LR={learning_rate}")

                        nn = NeuralNetwork(
                            inputSize=X_train.shape[1],
                            numClasses=y_train_encoded.shape[1],
                            hiddenLayersSize=hidden_layers_config,
                            neuronsPerLayer=neurons_per_layer_config,
                            learningRate=learning_rate,
                            method=method
                        )

                        for epoch in range(EPOCHS):
                            for x, y_enc in zip(X_train, y_train_encoded):
                                nn.train(x, y_enc)

                        y_pred_after = [nn.predictClass(x) for x in X_test]
                        acc_after = accuracy_score(y_test_int, y_pred_after)

                        print(f"Acurácia após treino: {acc_after*100:.2f}%")

                        plot_data.append({
                            'neurons': neurons_per_layer_config,
                            'layers': hidden_layers_config,
                            'accuracy': acc_after * 100,
                            'method': method,
                            'lr': learning_rate,
                            'perc_treino': perc
                        })

    df_plot = pd.DataFrame(plot_data)

    # --- Geração dos Gráficos 2D ---

    # Gráfico 1: Acurácia vs. Número de Neurônios (linhas = número de camadas)
    for method_name in df_plot['method'].unique():
        method_df = df_plot[df_plot['method'] == method_name]
        if method_df.empty:
            continue

        plt.figure(figsize=(10, 6))
        for num_camadas in sorted(method_df['layers'].unique()):
            camada_df = method_df[method_df['layers'] == num_camadas].sort_values(by='neurons')
            if not camada_df.empty:
                plt.plot(camada_df['neurons'], camada_df['accuracy'], marker='o', linestyle='-', label=f'{num_camadas} Camada(s)')

        plt.xlabel('Número de Neurônios por Camada')
        plt.ylabel('Acurácia (%)')
        plt.title(f'Acurácia vs. Neurônios - Método: {method_name}\n(LR={LEARNING_RATES[0]}, Treino={TRAINING_PERCENTAGES[0]*100}%)')
        plt.legend(title='Nº de Camadas')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.ylim(0, 100) # Garante que o eixo Y da acurácia vá de 0 a 100
        plt.tight_layout()
        plt.savefig(f"graficos/neural/2D_AccVsNeurons_Method_{method_name}.png")
        print(f"Gráfico 'Acurácia vs Neurônios' salvo para {method_name}")
        plt.close()

    # Gráfico 2: Acurácia vs. Número de Camadas (linhas = número de neurônios)
    for method_name in df_plot['method'].unique():
        method_df = df_plot[df_plot['method'] == method_name]
        if method_df.empty:
            continue

        plt.figure(figsize=(10, 6))
        for num_neuronios in sorted(method_df['neurons'].unique()):
            neuronio_df = method_df[method_df['neurons'] == num_neuronios].sort_values(by='layers')
            if not neuronio_df.empty:
                plt.plot(neuronio_df['layers'], neuronio_df['accuracy'], marker='s', linestyle='--', label=f'{num_neuronios} Neurônio(s)')

        plt.xlabel('Número de Camadas Ocultas')
        plt.ylabel('Acurácia (%)')
        plt.title(f'Acurácia vs. Camadas - Método: {method_name}\n(LR={LEARNING_RATES[0]}, Treino={TRAINING_PERCENTAGES[0]*100}%)')
        plt.legend(title='Nº de Neurônios/Camada')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.ylim(0, 100) # Garante que o eixo Y da acurácia vá de 0 a 100
        # Define os ticks do eixo X para serem apenas os inteiros testados para camadas
        plt.xticks(sorted(method_df['layers'].unique()))
        plt.tight_layout()
        plt.savefig(f"graficos/neural/2D_AccVsLayers_Method_{method_name}.png")
        print(f"Gráfico 'Acurácia vs Camadas' salvo para {method_name}")
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
            # Prevenção de overflow
            x = np.clip(x, -500, 500)
            return 1 / (1 + math.exp(-x))
        elif self.method == 'relu':
            return max(0, x)
        elif self.method == 'tanh':
            # Prevenção de overflow
            x = np.clip(x, -500, 500)
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
            # A derivada do ReLU é 1 para x > 0, e 0 caso contrário.
            # (self.lastOutput > 0) resulta em True (1) ou False (0).
            # .astype(float) converte para 1.0 ou 0.0.
            return (self.lastOutput > 0).astype(float) * error
        elif self.method == 'tanh':
            # Derivada da tanh(x) é 1 - tanh(x)^2.
            # self.lastOutput já é tanh(weightedSum).
            return (1 - self.lastOutput**2) * error
        else:
            raise ValueError("Método de ativação desconhecido para cálculo do delta. Use 'sigmoid', 'relu' ou 'tanh'.")

    def updateWeights(self, error):
        # CORREÇÃO: Usar calculateDelta para obter o delta correto conforme o método de ativação
        self.delta = self.calculateDelta(error)
        for i in range(self.inputSize):
            self.weights[i] += self.learningRate * self.delta * self.lastInputs[i]
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
        if len(inputs) != self.weights.shape[1]:
            raise ValueError("Input size must match weights size.")
    
        self.lastInputs = np.array(inputs)
        self.lastOutputs = []
        
        for i in range(len(self.weights)):
            weightedSum = np.dot(self.weights[i], inputs) + self.bias[i]
            # Prevenção de overflow para a sigmoide da camada de saída
            weightedSum = np.clip(weightedSum, -500, 500)
            output = 1 / (1 + math.exp(-weightedSum))
            self.lastOutputs.append(output)
        
        if not self.lastOutputs: # Caso de emergência, se lastOutputs estiver vazio
             return 1 # ou alguma classe padrão
        return self.lastOutputs.index(max(self.lastOutputs)) + 1

    def updateWeights(self, target):
        self.deltas = [self.lastOutputs[i] * (1 - self.lastOutputs[i] ) * (target[i] - self.lastOutputs[i])for i in range(self.numClasses)]
        for i in range(self.numClasses):
            self.weights[i] += self.learningRate * self.deltas[i] * self.lastInputs
        # Atualização do bias da camada de saída (estava faltando)
        for i in range(self.numClasses):
            self.bias[i] += self.learningRate * self.deltas[i]


class NeuralNetwork:
    def __init__(self, inputSize, numClasses, hiddenLayersSize=1, neuronsPerLayer=1, learningRate=0.01, method='sigmoid'):
        self.inputSize = inputSize
        self.numClasses = numClasses
        self.learningRate = learningRate
        # A camada de saída agora recebe 'neuronsPerLayer' como seu inputSize,
        # que é o número de saídas da última camada oculta.
        self.outputLayer = OutputLayer(neuronsPerLayer, numClasses, self.learningRate)
        self.hiddenLayers = []
        self.method = method
        self.createLayers(hiddenLayersSize, neuronsPerLayer)
    
    def createLayers(self, hiddenLayersSize, neuronsPerLayer):
        current_input_size = self.inputSize
        for i in range(hiddenLayersSize):
            layer = []
            for _ in range(neuronsPerLayer):
                layer.append(Perceptron(current_input_size, self.learningRate, self.method))
            self.hiddenLayers.append(layer)
            current_input_size = neuronsPerLayer # A entrada da próxima camada é a saída da atual

    def feedForward(self, inputs):
        current_inputs = np.array(inputs)
        for layer in self.hiddenLayers:
            outputs = []
            for neuron in layer:
                outputs.append(neuron.predict(current_inputs))
            current_inputs = np.array(outputs) # Saídas desta camada são entradas para a próxima
        return self.outputLayer.predict(current_inputs)
    
    def backpropagate(self, target):
        self.outputLayer.updateWeights(target)

        nextDeltas = self.outputLayer.deltas
        # nextWeights para a primeira etapa de retropropagação (da saída para a última camada oculta)
        # são os pesos da camada de saída.
        nextWeights = self.outputLayer.weights

        # Iterar pelas camadas ocultas de trás para frente
        for i in range(len(self.hiddenLayers) - 1, -1, -1):
            layer = self.hiddenLayers[i]
            currentDeltas = []
            errors_for_this_layer = [] # Erros que serão usados para atualizar os pesos desta camada

            for j, neuron in enumerate(layer):
                error_sum = 0
                # O erro para um neurônio j na camada oculta atual é a soma ponderada dos deltas da PRÓXIMA camada.
                # nextDeltas são os deltas da camada à frente (mais próxima da saída).
                # nextWeights[k][j] é o peso que conecta o neurônio j desta camada ao neurônio k da próxima camada.
                for k in range(len(nextDeltas)): # k itera sobre os neurônios da PRÓXIMA camada
                    error_sum += nextDeltas[k] * nextWeights[k][j]
                
                # O neurônio calcula seu próprio delta e atualiza seus pesos
                # A função updateWeights do Perceptron já calcula o delta interno usando calculateDelta(error_sum)
                neuron.updateWeights(error_sum)
                currentDeltas.append(neuron.delta) # Guarda o delta calculado por este neurônio
            
            # Para a próxima iteração (a camada anterior a esta),
            # os 'nextDeltas' serão os 'currentDeltas' que acabamos de calcular.
            nextDeltas = currentDeltas
            # E os 'nextWeights' serão os pesos dos neurônios da camada atual.
            nextWeights = np.array([neuron.weights for neuron in layer])


    def train(self, inputs, target):
        inputs_array = np.array(inputs)
        target_array = np.array(target)
        if len(inputs_array) != self.inputSize:
            raise ValueError("Input size must match the network input size.")
        
        self.feedForward(inputs_array) # feedForward para popular lastOutputs e lastInputs
        self.backpropagate(target_array)

    def predictClass(self, inputs):
        return self.feedForward(inputs)




if __name__ == "__main__":
    # Certifique-se que o arquivo CSV está no caminho correto ou ajuste o path.
    # Exemplo: testar_neural_network("dataset/treino_sinais_vitais_com_label.csv")
    # Se o arquivo não existir, esta chamada irá falhar.
    # Crie um arquivo dummy CSV se necessário para testar a estrutura do código.
    # Exemplo de criação de dummy dataset para teste:
    # if not os.path.exists("dataset/treino_sinais_vitais_com_label.csv"):
    #     print("Arquivo de dataset não encontrado. Criando um dummy para teste.")
    #     os.makedirs("dataset", exist_ok=True)
    #     dummy_data = np.random.rand(100, 5) # 4 features, 1 label
    #     dummy_data[:, -1] = np.random.randint(1, 5, 100) # Labels 1, 2, 3, 4
    #     dummy_df = pd.DataFrame(dummy_data, columns=['f1', 'f2', 'f3', 'f4', 'label'])
    #     dummy_df.to_csv("dataset/treino_sinais_vitais_com_label.csv", index=False)
        
    testar_neural_network("dataset/treino_sinais_vitais_com_label.csv")
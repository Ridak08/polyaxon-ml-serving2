import joblib
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

def train_and_eval_cnn(
    test_size=0.2,
    random_state=1012,
    model_path=None,
):
    columnas = ['source_port', 'destination_port', 'protocol', 'packets', 'length','fin_flag', 'syn_flag', 'rst_flag','psh_flag', 'ack_flag', 'urg_flag','cwe_flag', 'ece_flag', 'Label']
    df_shuffle = pd.read_csv("./df_5000_final_14_caracteristicas.csv", usecols = columnas, skipinitialspace=True)
    features = df_shuffle.drop("Label", axis=1).values
    labels = df_shuffle["Label"].values
    
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=random_state)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = X_train.reshape(-1, features.shape[1], 1)
    X_test = X_test.reshape(-1, features.shape[1], 1)

    def conv1d(input_data, kernel, stride=1, padding='same'):
        # Calculate output shape
        output_shape = (input_data.shape[0], (input_data.shape[1] - kernel.shape[0]) // stride + 1, kernel.shape[1])
        
        # Initialize output array
        output = np.zeros(output_shape)
        
        # Perform convolution
        for i in range(output_shape[1]):
            for j in range(kernel.shape[0]):
                output[:, i, :] += input_data[:, i*stride:i*stride+kernel.shape[0], :] * kernel[j, :]
        
        return output

    def max_pooling(input_data, pool_size=2):
        # Calculate output shape
        output_shape = (input_data.shape[0], input_data.shape[1] // pool_size, input_data.shape[2])
        
        # Initialize output array
        output = np.zeros(output_shape)
        
        # Perform max pooling
        for i in range(output_shape[1]):
            output[:, i, :] = np.max(input_data[:, i*pool_size:i*pool_size+pool_size, :], axis=1)
        
        return output

    def flatten(input_data):
        return input_data.reshape(input_data.shape[0], -1)

    def dense(input_data, weights, bias):
        return np.dot(input_data, weights) + bias

    def sgd(model, X, y, learning_rate=0.01):
        # Calculate loss
        loss = np.mean((model.predict(X) - y) ** 2)
        
        # Calculate gradients
        gradients = []
        for layer in model.layers:
            gradients.append(np.dot(layer.input_data.T, (model.predict(X) - y)))
        
        # Update weights
        for i, layer in enumerate(model.layers):
            layer.weights -= learning_rate * gradients[i]
        
        return loss

    class CNN:
        def __init__(self):
            self.layers = []

        def add_conv1d(self, num_filters, kernel_size, strides, padding, activation):
            self.layers.append({'type': 'conv1d', 'num_filters': num_filters, 'kernel_size': kernel_size, 'strides': strides, 'padding': padding, 'activation': activation})

        def add_max_pooling(self, pool_size):
            self.layers.append({'type': 'max_pooling', 'pool_size': pool_size})

        def add_flatten(self):
            self.layers.append({'type': 'flatten'})

        def add_dense(self, num_units, activation):
            self.layers.append({'type': 'dense', 'num_units': num_units, 'activation': activation})

        def predict(self, X):
            output = X
            for layer in self.layers:
                if layer['type'] == 'conv1d':
                    output = conv1d(output, np.random.rand(layer['num_filters'], layer['kernel_size'], 1), layer['strides'], layer['padding'])
                    output = np.maximum(output, 0) if layer['activation'] == 'relu' else output
                elif layer['type'] == 'max_pooling':
                    output = max_pooling(output, layer['pool_size'])
                elif layer['type'] == 'flatten':
                    output = flatten(output)
                elif layer['type'] == 'dense':
                    output = dense(output, np.random.rand(output.shape[1], layer['num_units']), np.zeros(layer['num_units']))
                    output = np.maximum(output, 0) if layer['activation'] == 'relu' else output
            return output

    model_cnn = CNN()
    model_cnn.add_conv1d(32, kernel_size=4, strides=2, padding='same', activation='relu')

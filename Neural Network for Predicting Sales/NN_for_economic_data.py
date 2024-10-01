import sys
import numpy as np
import pandas as pd
import math
import random
import os
from sklearn.preprocessing import StandardScaler


class Network:
    def __init__ (self, layers):
        #layers is a list of layer instances, forming the MLP
        self.layers = layers

    def forward_pass_network (self, input_data):
        activations = input_data
        for layer in self.layers:
            activations = layer.forward_pass_layer(activations)
        return activations
        
    def backprop_network (self, first_upstream_gradient):
        upstream_gradient = first_upstream_gradient
        for layer in reversed(self.layers):
            upstream_gradient = layer.backprop_layer(upstream_gradient)
        
    def update_weights_network (self, learning_rate):
        for layer in self.layers:
            layer.update_weights_layer(learning_rate)

    def train (self, input_data, ideal_y):
        actual_y = self.forward_pass_network(input_data)
        first_upstream_grad = 2 * np.subtract(actual_y, ideal_y)
        self.backprop_network(first_upstream_grad)
        self.update_weights_network(0.1)


class Layer:
    def __init__ (self, weights):
        self.weights = weights
        self.z = np.array([])
        self.incoming_activations_hat = np.array([])
        self.grad_of_E_wrt_w = np.array([])

    def forward_pass_layer (self, incoming_activations):
        self.incoming_activations_hat = np.vstack((incoming_activations, [[1]]))
        for row in self.incoming_activations_hat:
            for column in row:
                if math.isnan(column):
                    print("a value in incoming activations hat is nan")
                    sys.exit()
        for row in self.weights:
            for column in row:
                if math.isnan(column):
                    print("a value in weights is nan")
                    sys.exit()
        self.z = np.matmul(self.weights, self.incoming_activations_hat)
        y = relu(self.z)
        return y

    def backprop_layer (self, upstream_grad, max_norm=1.0):
        h_prime_z = relu_derivative(self.z)
        grad_of_E_wrt_z = np.multiply(h_prime_z, upstream_grad)
        t = self.incoming_activations_hat.transpose()
        self.grad_of_E_wrt_w = np.matmul(grad_of_E_wrt_z, t)

        # Compute gradient norm for clipping
        grad_norm = np.linalg.norm(self.grad_of_E_wrt_w)

        # Clip gradients if norm exceeds the threshold
        if grad_norm > max_norm:
            self.grad_of_E_wrt_w = self.grad_of_E_wrt_w * (max_norm / grad_norm)

        w_t = self.weights.transpose()
        grad_of_E_wrt_xhat = np.matmul(w_t, grad_of_E_wrt_z)
        next_upstream_grad = np.delete(grad_of_E_wrt_xhat, grad_of_E_wrt_xhat.size - 1, 0)
        
        return next_upstream_grad

    def update_weights_layer (self, learning_rate):
        self.weights = np.subtract(self.weights, learning_rate * self.grad_of_E_wrt_w)

def relu_unvectorized (element):
    if element >= 0:
        return element
    else:
        return 0.1 * element
relu = np.vectorize(relu_unvectorized)

def relu_derivative_unvectorized (element):
    if element >= 0:
        return 1
    else:
        return 0.1
relu_derivative = np.vectorize(relu_derivative_unvectorized)

def he_initialization(shape):
    return np.random.randn(*shape) * np.sqrt(2 / shape[0])


weights1 = he_initialization((64, 27))
weights2 = he_initialization((16, 65))
weights3 = he_initialization((1, 17))


array_of_layers = [Layer(weights1), Layer(weights2), Layer(weights3)]
my_network = Network(array_of_layers)


train_data = pd.read_csv("./train-1992-2017.csv")
test_data = pd.read_csv("./test-2018-2024.csv")
test_ideal_outputs = pd.read_csv("./true_sales@t+1.csv")


features = ["index", "GDP", "M2V", "FD-inc-tax-withheld", "PMSAVE", "CPALTT01USM657N", "UMCSENT", "`x", "UNEMPLOY", "CP", "USTRADE", "INDPRO", "USTPU", "CSCICP03USM665S", "CRDQUSAPABIS", "PCECC96", "CUSR0000SETA01", "DPI", "FEDFUNDS_x", "FYFSGDA188S", "AWHMAN", "CIVPART_y", "OPHPBS", "HOUST", "PCE", "PAYEMS"]
X = train_data[features]
X_test = test_data[features]
y = train_data[["sales@t+1"]]
y_test = test_ideal_outputs[["sales@t+1"]]


scaler_x = StandardScaler()
#scaler_y = StandardScaler()
X_normalized = scaler_x.fit_transform(X)
X_test_normalized = scaler_x.transform(X_test)


# Convert back to DataFrame and retain column names
X_normalized_df = pd.DataFrame(X_normalized, columns = X.columns)
X_test_normalized_df = pd.DataFrame(X_test_normalized, columns = X_test.columns)


epochs = 250
for epoch in range(epochs):
    for i in range(y.size):
        input_data = X_normalized_df.iloc[i].to_numpy().reshape(-1, 1)
        ideal_output = y.iloc[i].to_numpy().reshape(-1, 1)
        my_network.train(input_data, ideal_output)

sum_error = 0
for i in range (y_test.size):
    estimated = my_network.forward_pass_network(X_test_normalized_df.iloc[i].to_numpy().reshape(-1, 1))[0][0]
    actual = (y_test.iloc[i])[0]
    sum_error += ((abs(estimated - actual)) / actual) * 100

average_percent_error = sum_error / y_test.size
print(average_percent_error)



import numpy as np

inputs = [0.5, 0.9, 0.2]

weights = [5.0, 3.0, 1.0]
bias = 0.3

suma_ponderada = np.dot(inputs, weights) + bias

def sigmoide(x) : return 1 / (1 + np.exp(-x))
print(sigmoide(suma_ponderada))

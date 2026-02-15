import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])

modelo = Sequential()

modelo.add(Input(shape=(2,))) 

modelo.add(Dense(16, activation='tanh'))
modelo.add(Dense(16, activation='tanh'))
modelo.add(Dense(1, activation='sigmoid'))

optimizador = Adam(learning_rate=0.01)

modelo.compile(loss='binary_crossentropy', optimizer=optimizador, metrics=['accuracy'])

print("Entrenando red neuronal (Estrategia Tanh)...")
modelo.fit(X, y, epochs=5000, verbose=0)

print("\nResultados finales del XOR:")
prediccion = modelo.predict(X)

print("Valores crudos:\n", prediccion)
print("Redondeados:\n", prediccion.round())
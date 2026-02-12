import numpy as np
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

X = np.array([
    [0.1, 0.1, 0.1],
    [0.9, 0.8, 0.9],
    [0.1, 0.2, 0.1], 
    [0.9, 0.9, 0.8]
])
y = np.array([0, 1, 0, 1])

modelo = Sequential()
modelo.add(Dense(units=1, input_shape=[3], activation= 'sigmoid'))
modelo.compile(optimizer='adam', loss='mean_squared_error')

modelo.fit(X, y, epochs = 3000, verbose=0)

nuevo_dato = np.array([[0.8, 0.8, 0.8]])
prediccion = modelo.predict(nuevo_dato)
print(f"Predicción para entrada alta: {prediccion[0][0]:.4f}")
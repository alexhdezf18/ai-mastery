import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

modelo = Sequential()

modelo.add(Dense(units=1, input_shape=[3], activation= 'sigmoid'))
modelo.compile(optimizer='sgd', loss='mean_squared_error')
print(modelo.summary())
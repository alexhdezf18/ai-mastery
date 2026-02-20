import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

print("Cargando dataset...")
fashion_mnist = tf.keras.datasets.fashion_mnist
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

X_train = X_train / 255.0
X_test = X_test / 255.0

X_train.reshape(-1, 28, 28, 1)                                                                                       
X_test.reshape(-1, 28, 28, 1)                                                                                       

print(f"Forma de los datos de entrenamiento: {X_train.shape}")
print(f"Forma de los datos de prueba: {X_test.shape}")
print(f"Cantidad de clases (tipos de ropa): {len(np.unique(y_train))}")

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

index = 0  
plt.figure(figsize=(3,3))
plt.title(f"Etiqueta: {y_train[index]} -> {class_names[y_train[index]]}")
plt.colorbar()
plt.show()
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix   
from sklearn.metrics import ConfusionMatrixDisplay


horas = np.random.randint(0, 21, size=100)
probs = horas / 25 
exito = np.random.binomial(n=1, p=probs)


X = horas.reshape(-1, 1)
y = exito

modelo = LogisticRegression()
modelo.fit(X, y)

y_pred = modelo.predict(X)
matriz = confusion_matrix(exito, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=matriz)
disp.plot()
plt.show()

plt.scatter(horas, exito, color='blue', label='Datos Reales')

x_suave = np.linspace(0, 20, 1000).reshape(-1, 1)

y_probabilidad = modelo.predict_proba(x_suave)[:, 1]

plt.plot(x_suave, y_probabilidad, color='red', label='Curva Logística (S)')
plt.legend()
plt.show()


data = np.array([2, 9, 11, 19]).reshape(-1, 1)
probabilidad = modelo.predict_proba(data)[:, 1]
print(probabilidad)


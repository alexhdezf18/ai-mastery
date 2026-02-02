import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

modelo_cargado = joblib.load('modelo_prueba.joblib')

df = pd.read_csv('datos.csv')

X = df[['x']]
y = df['y']

predicciones = modelo_cargado.predict(X)

error_mae = mean_absolute_error(y, predicciones)
print(f"Error Promedio (MAE): {error_mae}")

df['ventas_estimadas'] = predicciones

df.to_csv('predicciones_finales.csv', index=False)

plt.scatter(X, y, color='blue')
plt.plot(X, predicciones, color='red')
plt.show()
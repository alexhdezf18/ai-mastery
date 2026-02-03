import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

df = pd.read_csv('carwash_data.csv')

X = df[['temperatura', 'fin_de_semana']]
y = df['clientes']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modelo = LinearRegression()
modelo.fit(X_train, y_train)

print(f"Coeficientes descubiertos: {modelo.coef_}")
print(f"Intercepto base: {modelo.intercept_}")

y_pred = modelo.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)

print(f"Error Promedio Absoluto (MAE): {mae}")

df2 = pd.DataFrame({
    'temperatura': [32],
    'fin_de_semana': [1],
})

prediction = modelo.predict(df2)

print(f"Para un sábado a 32°C, esperamos {prediction[0]:.0f} clientes")
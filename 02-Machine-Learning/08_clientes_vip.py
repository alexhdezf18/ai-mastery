import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib

opciones = ['Toyota', 'Ford', 'BMW', 'Audi']
marcas = np.random.choice(opciones, size=100)
antiguedad = np.random.randint(1, 16, size=100)

marcas_vip = np.isin(marcas, ['BMW', 'Audi'])

vip = np.where(marcas_vip, 1, 0)

df = pd.DataFrame({
    'marcas' : marcas,
    'antiguedad' : antiguedad,
    'vip' : vip,
})

traductor = LabelEncoder()

df['marca_numerica'] = traductor.fit_transform(df['marcas'])
print(df.head())

X = df[['marca_numerica', 'antiguedad']]
y = df['vip']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
modelo = LogisticRegression()
modelo.fit(X_train, y_train)

marca_codificada = traductor.transform(['Ford'])
ford = pd.DataFrame(
    [[marca_codificada[0], 5]], 
    columns=['marca_numerica', 'antiguedad'] 
)
prediccion = modelo.predict(ford)
print(prediccion)

marca_codificada_audi = traductor.transform(['Audi'])
audi = pd.DataFrame(
    [[marca_codificada_audi[0], 15]], 
    columns=['marca_numerica', 'antiguedad'] 
)
print(modelo.predict_proba(audi))

joblib.dump(modelo, 'modelo_vip.joblib')
joblib.dump(traductor, 'traductor_marcas.joblib')
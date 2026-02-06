import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

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



import pandas as pd
import numpy as np

days = 365
temp = np.random.randint(10, 36, size=days)
weekend =np.random.choice([0, 1], size=days, p=[0.7, 0.3])
noise = np.random.randint(-5, 6, size=days)

customers = 10 + (2 * temp) + (50 * weekend) + noise

df = pd.DataFrame({
    'dia': range(1, days + 1),
    'temperatura': temp,
    'fin_de_semana': weekend,
    'clientes': customers
})

df.to_csv('carwash_data.csv', index=False)
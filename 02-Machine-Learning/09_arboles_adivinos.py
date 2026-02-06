from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree

X = [[10, 1], [80, 1], [10, 0], [80, 0]]
y = [1, 0, 0, 0]

modelo = DecisionTreeClassifier()
modelo.fit(X, y)

plot_tree(modelo, feature_names=['Edad', 'Futbol'], class_names=['Casa', 'Estadio'], filled=True)
plt.show()
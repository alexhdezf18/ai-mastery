from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot as plt
from sklearn.tree import plot_tree

dataset = load_iris()
X = dataset.data
y = dataset.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modelo = DecisionTreeClassifier(max_depth=3)
modelo.fit(X_train, y_train)

precision = modelo.score(X_test, y_test)
print(f"Precisión del modelo podado: {precision:.2%}")

plt.figure(figsize=(15, 10))

plot_tree(
    modelo, 
    feature_names=dataset.feature_names, 
    class_names=dataset.target_names, 
    filled=True
)
plt.show()
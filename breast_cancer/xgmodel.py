import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,accuracy_score
from xgboost import XGBClassifier
from xgboost import plot_importance
import matplotlib.pyplot as plt
import numpy as np
from utils import *

data = load_breast_cancer()

df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target

y = df["target"]
X = df.drop("target", axis=1)

# Used the following code to draw the graphs
'''model = XGBClassifier()
model.fit(X,y)

draw_bar_graph(model,X,y)

plot_importance(model)
plt.show()'''

X = X.drop(["mean perimeter","concave points error","mean fractal dimension","perimeter error","worst compactness","texture error","mean radius","radius error","smoothness error","worst fractal dimension","concavity error","mean compactness","mean concavity","mean area","mean symmetry","symmetry error","fractal dimension error"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

model = XGBClassifier()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_true=y_test,y_pred=y_pred)
f1 = f1_score(y_true=y_test,y_pred=y_pred)
print(f"Accuracy = {(accuracy * 100)} and F1 = {f1 * 100}")

plot_xg_graph(model,y_train,y_test,X)

# Fazer previsões de probabilidade para os dados de teste
y_probs = model.predict_proba(X_test)

import random

# Selecionar instâncias onde a probabilidade da classe 1 é alta
indices_classe_1 = np.where(y_probs[:, 0] > 0.9)[0]

# Selecionar um índice aleatório entre os índices onde a previsão é 1
indice_aleatorio = random.choice(indices_classe_1)

# Exibir os valores correspondentes ao índice aleatório
print("Valores que geram um resultado 1:")
print(f"Índice: {indice_aleatorio}, Valores: {X_test.iloc[indice_aleatorio]}")



model.save_model('modelo_xgboost.xgb')

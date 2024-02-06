import tkinter as tk
from tkinter import ttk
from sklearn.datasets import load_breast_cancer
from xgboost import XGBClassifier
import xgboost as xgb

data = load_breast_cancer()

model = XGBClassifier()
model.load_model('modelo_xgboost.xgb')

def do_predictions():
    values = [float(entry.get()) for entry in entries]
    
    previsao = model.predict([values])
    
    result_label.config(text=f'Prediction: {previsao[0]}')

root = tk.Tk()
root.title("Breast Cancer Detection")

features = ['mean texture', 'mean smoothness', 'mean concave points', 'area error',
       'compactness error', 'worst radius', 'worst texture', 'worst perimeter',
       'worst area', 'worst smoothness', 'worst concavity',
       'worst concave points', 'worst symmetry']

entries = []
for i, feature_name in enumerate(features):
    label = ttk.Label(root, text=feature_name)
    label.grid(row=i, column=0, padx=10, pady=5, sticky="e")
    entry = ttk.Entry(root)
    entry.grid(row=i, column=1, padx=10, pady=5)
    entries.append(entry)


prever_button = ttk.Button(root, text="Do Prediction", command=do_predictions)
prever_button.grid(row=len(entries)+1, columnspan=2, padx=10, pady=10)

result_label = ttk.Label(root, text="")
result_label.grid(row=len(entries)+2, columnspan=2, padx=10, pady=5)

root.mainloop()

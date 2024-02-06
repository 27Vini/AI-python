from xgboost import XGBClassifier
from xgboost import plot_importance
import matplotlib.pyplot as plt

def draw_bar_graph(model,X,y):
    importance = model.feature_importances_
    feature_names = X.columns

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(importance)), importance, tick_label=feature_names)
    plt.xticks(rotation=90)
    plt.xlabel('Feature')
    plt.ylabel('Importance')
    plt.title('Features Importance')
    plt.show()
   
def plot_xg_graph(model,y_train,y_test,X):
    plt.scatter(range(len(y_train)), y_train, color='blue', label='y_train')
    plt.scatter(range(len(y_test)), y_test, color='red', label='y_test')

    y_pred_all = model.predict(X)

    plt.plot(range(len(y_pred_all)), y_pred_all, color='green', linestyle='-', linewidth=2, label='XGBoost Model')

    plt.legend()
    plt.xlabel('Data Index')
    plt.ylabel('Class')
    plt.title('Y_train and y_test dots with XGBoost line')
    plt.show()
 
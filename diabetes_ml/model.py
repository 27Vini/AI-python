import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score,f1_score

df = pd.read_csv("diabetes.csv")
features = [x for x in df.columns if x not in "Outcome"]

'''
code to get the best values
param_grid = {
    'learning_rate': [0.1, 0.01, 0.001],
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5]
}
grid_search = GridSearchCV(estimator=xgb_classifier, param_grid=param_grid, cv=5, scoring='accuracy')

grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
print("Melhores hiperparâmetros:", best_params)'''

X_train,X_test, y_train,y_test = train_test_split(df[features],df["Outcome"],test_size=0.3,random_state=42)

xgb_classifier = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.01)

cv_scores = cross_val_score(xgb_classifier, X_train, y_train, cv=5, scoring='accuracy')

print("Cross validation scores: ", cv_scores)

mean_cv_score = cv_scores.mean()
print("Cross validation mean:", mean_cv_score)
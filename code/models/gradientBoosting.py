import pandas as pd
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, train_test_split
from xgboost import XGBClassifier


__author__ = 'antoine1.adam[at]etudiant.univ-rennes2.fr'

X = pd.read_json("../../data/train5k.json").setIndex("Id")
y = pd.read_csv('../../data/train_label.csv', index_col='Id', dtype={'Category': 'category'})

dtype = pd.SparseDtype(float, fill_value=0.)
X = X.astype(dtype)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/2, random_state=42069)


params = {'n_estimators': [100, 200, 300, 400, 500], 
          'learning_rate': [.001, .01, .1, .2, .5]}

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42067)
clf = GridSearchCV(XGBClassifier(), 
                   param_grid=params,
                   n_jobs=-1,
                   verbose=1)
clf.fit(X_train, y_train.to_numpy().ravel())

print("score sur le jeu de données test : ", clf.score(X_test, y_test))
print("*"*20)
print("paramètres : ", clf.best_params_)
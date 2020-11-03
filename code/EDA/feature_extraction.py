# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 11:41:09 2020

@author: antoineadam

Remarque importante : ne pas oublier de dl les données nltk en local !!
"""
import sys
sys.path.append("..")

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2

from utils import clean_text

data = pd.read_json('../../data/train.json').set_index('Id')
labels = pd.read_csv('../../data/train_label.csv', index_col='Id', dtype={'Category': 'category'})
desc = data.loc[:, 'description']
gender = data.rename(columns={'gender': 'Gender'}).loc[:, 'Gender'].astype('category')
gender = pd.get_dummies(gender, drop_first=True)
del data

# Vectorisation
vectorizer = TfidfVectorizer(preprocessor=clean_text)
X = vectorizer.fit_transform(desc)
del desc

#Selection des K meilleurs
select_f_classif = SelectKBest(k = 5000)
f_classif = select_f_classif.fit_transform(X, labels.to_numpy().ravel())
del X

# Regroupement
names = np.array(vectorizer.get_feature_names())
mask = select_f_classif.get_support()
features = names[mask].tolist()
# Ci dessus, je transforme en np.array pour pouvoir plus facilement utiliser le masque booléen du selectKbest
tfidf = pd.concat([
    pd.DataFrame.sparse.from_spmatrix(f_classif, columns=features),
    gender], axis=1)
del f_classif
del gender
del labels

dtype = pd.SparseDtype(float, fill_value=0.)
tfidf = tfidf.astype(dtype)

if __name__ == "__main__":  
    # à utiliser seulement quand on change le résultat.
    tfidf.to_json("../../data/train5k.json") 
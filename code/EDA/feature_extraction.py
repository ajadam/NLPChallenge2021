# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 11:41:09 2020

@author: antoineadam

Remarque importante : ne pas oublier de dl les données nltk en local !!
"""
import pandas as pd
import numpy as np

import string
import re

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2



wnl = WordNetLemmatizer()
sw = set(stopwords.words('english'))
space = re.compile('[/(){}\[\]\|@,;]')
symbols = re.compile('[^0-9a-z #+_]')

def clean_text(text):
    text = symbols.sub(' ', space.sub(' ', text.lower()))
    text = re.sub(" +", " ", re.sub(r'\d+', ' ', re.sub(rf'[{string.punctuation}]', '', text)))
    tokens = word_tokenize(text)
    tokens = [tok for tok in tokens if tok not in sw]
    tokens = [wnl.lemmatize(tok) for tok in tokens]
    return " ".join(tokens)


data = pd.read_json('../../data/train.json').set_index('Id')
labels = pd.read_csv('../../data/train_label.csv', index_col=0, dtype = 'category')
desc = data.loc[:, 'description']
gender = data.loc[:, 'gender']
del data

# Vectorisation
vectorizer = TfidfVectorizer(preprocessor=clean_text)
X = vectorizer.fit_transform(desc)
del desc

#Selection des K meilleurs
"""
Explication du "labels.to_numpy().ravel()" :

la méthode `.fit()` souhaite un array de forme y.shape = (n, ) : https://stackoverflow.com/questions/34165731/a-column-vector-y-was-passed-when-a-1d-array-was-expected
i.e. un array applati. De plus plutôt que `pd.df.values`, il vaut mieux utiliser `pd.df.to_numpy` : https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_numpy.html#pandas.DataFrame.to_numpy, 
enfin utiliser 'np.ravel()' https://numpy.org/doc/stable/reference/generated/numpy.ravel.html pour applatir l'array.
"""
select_f_classif = SelectKBest(k = 1000)
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

if __name__ == "__main__":
    print(features)
    print("")
    print(tfidf)
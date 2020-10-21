# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 11:41:09 2020

@author: antoineadam

Remarque importante : ne pas oublier de dl les données nltk en local !!
"""
import pandas as pd
import dask.dataframe as dd

import string
import re

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer



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
desc_clean = data.description.map(clean_text)
gender = data.gender
del data
# Vectorisation
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(desc_clean)

data_final = pd.concat([
    pd.DataFrame.sparse.from_spmatrix(X, columns=vectorizer.get_feature_names()),
    gender
], axis = 1)
del desc_clean
del X
del gender

if __name__ == "__main__":
    print(data_final.head())
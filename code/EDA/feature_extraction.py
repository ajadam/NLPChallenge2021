#!/usr/bin/env python3
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

from sklearn.feature_extraction.text import CountVectorizer



wnl = WordNetLemmatizer()
sw = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(rf'[{string.punctuation}]', '', text)
    tokens = word_tokenize(text)
    tokens = [tok for tok in tokens if tok not in sw]
    tokens = [wnl.lemmatize(tok) for tok in tokens]
    return " ".join(tokens)


data = pd.read_json('../../data/train.json').set_index('Id')

"""
Prendre un nb de partitions adéquat
"""
ddata = dd.from_pandas(data, npartitions = 8)

desc = ddata.loc[:, 'description'].copy()
desc_clean = desc.apply(clean_text, meta=('description', 'str'))

# Vectorisation
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(desc_clean)
matrice_desc = X.toarray()


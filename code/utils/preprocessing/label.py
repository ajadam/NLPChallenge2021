# -*- coding: utf-8 -*-

__all__ = [
    "TopicLDA"
]

import numpy as np
import pandas as pd
from gensim import corpora, models
from sklearn.exceptions import NotFittedError

from base import clean_text

class TopicLDA:
    """
    Latent Dirichlet Allocation
    Sorts the document according to its topic.
    """

    def __init__(self, num_labels):
        self.num_labels = num_labels
        self.dictionary, self.model = None, None
    
    def fit(self, X, lemmatize = False, stem = False):
        """
        X: vector of str
        *: clean_text args
        """
        X = pd.Series(X) if not isinstance(X, pd.Series) else X
        X = X.apply(clean_text, args = (lemmatize, stem)).str.split()
        self.dictionary = corpora.Dictionary(X)
        bow = map(self.dictionary.doc2bow, X)
        self.model = models.ldamodel.LdaModel(
            bow, num_topics = self.num_labels,
            id2word = self.dictionary,
            passes = 50, minimum_probability = 0
        )
    
    def predict(self, X, lemmatize = False, stem = False):
        """
        X: vector of str
        :return: vector of logits, shape: (len(X) num_labels)
        """
        if not (self.dictionary and self.model):
            raise NotFittedError('No fitted vocabulary')
        X = pd.Series(X) if not isinstance(X, pd.Series) else X
        X = X.apply(clean_text, args = (lemmatize, stem)).str.split()
        logits = self.model[map(self.dictionary.doc2bow, X)]
        return np.array([[x[1] for x in doc] for doc in logits])
    
    def fit_predict(self, X, lemmatize = False, stem = False):
        X = pd.Series(X) if not isinstance(X, pd.Series) else X
        X = X.apply(clean_text, args = (lemmatize, stem)).str.split()
        self.dictionary = corpora.Dictionary(X)
        bow = list(map(self.dictionary.doc2bow, X))
        self.model = models.ldamodel.LdaModel(
            bow, num_topics = self.num_labels,
            id2word = self.dictionary,
            passes = 50, minimum_probability = 0
        )
        logits = self.model[bow]
        return np.array([[x[1] for x in doc] for doc in logits])
    
    def __repr__(self):
        if self.model:
            return str(self.model.print_topics(self.num_labels))
        else:
            raise NotFittedError('No fitted vocabulary')
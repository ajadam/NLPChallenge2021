import string
import re
import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

from gensim import corpora, models
from sklearn.exceptions import NotFittedError


wnl = WordNetLemmatizer()
stemmer = PorterStemmer()
sw = set(stopwords.words('english'))
space = re.compile('[/(){}\[\]\|@,;]')
symbols = re.compile('[^0-9a-z #+_]')


def clean_text(text, lemmatization = True, stemming=True):
    """
    Clean text
    text: (str)
    lemmatization: (bool)
    :return: (str) cleaned text
    """
    global wnl, sw, space, symbols, stemmer
    text = symbols.sub(' ', space.sub(' ', text.lower()))
    text = re.sub(" +", " ", re.sub(r'\d+', ' ', re.sub(rf'[{string.punctuation}]', ' ', text)))
    tokens = word_tokenize(text)
    tokens = [tok for tok in tokens if tok not in sw]
    tokens = list(map(wnl.lemmatize, tokens)) if lemmatization else tokens
    tokens = list(map(stemmer.stem, tokens)) if stemming else tokens
    return " ".join(tokens)


class TopicLDA:
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


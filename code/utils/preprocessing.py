import string
import re

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


wnl = WordNetLemmatizer()
sw = set(stopwords.words('english'))
space = re.compile('[/(){}\[\]\|@,;]')
symbols = re.compile('[^0-9a-z #+_]')


def clean_text(text, lemmatization = True):
    """
    Clean text
    text: (str)
    lemmatization: (bool)
    :return: (str) cleaned text
    """
    global wnl, sw, space, symbols
    text = symbols.sub(' ', space.sub(' ', text.lower()))
    text = re.sub(" +", " ", re.sub(r'\d+', ' ', re.sub(rf'[{string.punctuation}]', ' ', text)))
    tokens = word_tokenize(text)
    tokens = [tok for tok in tokens if tok not in sw]
    tokens = list(map(wnl.lemmatize, tokens)) if lemmatization else tokens
    return " ".join(tokens)
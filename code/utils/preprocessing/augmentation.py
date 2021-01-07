# -*- coding: utf-8 -*-

__all__ = [
    "BeyondBackTranslator",
    "GenderSwap"
]

import pandas as pd
import numpy as np
import boto3
from sklearn.preprocessing import MultiLabelBinarizer
from gensim.models import Word2Vec
from gensim import downloader as api
from nltk.tokenize import word_tokenize


class BeyondBackTranslator:
    """
    Beyond Back Translation
    """
    def __init__(self, source, intermediate):
        """
        source: (str), lang ISO 639-1
        intermediate: (str), lang ISO 639-1
        """
        self.source = source
        self.target = intermediate
        self.client = boto3.client('translate')
        self.oversampled = None
        self.forward, self.generated = None, None

        
    def translate_(self, data, source, target):
        """
        data: (iterable), [str,]
        source: str, lang ISO 639-1
        target: str, lang ISO 639-1
        :return: iterable
        """
        totarget = list()
        message = "Translate from {} to {}: {}/{}"
        for i, text in enumerate(data):
            print(message.format(source, target, i, len(data)), end='\r', flush = True)
            totarget.append(self.client.translate_text(
                Text = text,
                SourceLanguageCode = source,
                TargetLanguageCode = target
            )['TranslatedText'])
        return totarget

    
    def compute(self, X, sX, y):
        """
        Compute stats for oversampling X by sX
        """
        self.binarizer = MultiLabelBinarizer()
        bin_sX = self.binarizer.fit_transform(sX)
        self.classA, self.classB = self.binarizer.classes_
        
        raw = pd.concat([
            pd.DataFrame(bin_sX).reset_index(drop=True),
            y.reset_index(drop=True)
        ], axis=1)
        stats = raw.groupby(
            raw.columns[2], as_index=False).agg({
                0: 'sum', 1: 'sum'
        })
        
        stats_dict = stats.assign(
            low = stats[0] > stats[1]
        ).assign(
            low = lambda x:x.low.map(
                lambda x:self.classB if x else self.classA)
        ).to_dict(orient='records')
        
        return stats_dict
    
    
    def oversampling(self, X, sX, y):
        """
        Oversampling data by sX variable
        """
        select = pd.concat([
            X.reset_index(drop=True),
            sX.reset_index(drop=True),
            y.reset_index(drop=True)
        ], axis=1)
        stats_dict = self.compute(X, sX, y)
        p = []
        i = 1
        
        for d in stats_dict:
            print(f'sampling Category {i}/{len(stats_dict)}', end='\r', flush=True)
            if d['low'] == self.classA:
                mult = d[0]*2
                if mult < d[1]:
                    n = d[1] - mult if d[1] - mult >= mult else int(d[0]/2)
                    df1 = select.query(f"{sX.name} == '{self.classA}' and {y.name} == '{d[f'{y.name}']}'").sample(frac=4/5)
                    df2 = select.query(f"{sX.name} == '{self.classB}' and {y.name} == '{d[f'{y.name}']}'").sample(n=n)
                    df1['swap'], df2['swap'] = False, True
                    df = pd.concat([df1, df2])
                else:
                    n = (mult-d[1])/(d[1]*1.1)
                    df = select.query(f"{sX.name} == '{self.classA}' and {y.name} == '{d[f'{y.name}']}'").sample(frac = n)
                    df['swap'] = False
            else:
                mult = d[1]*2
                if mult < d[0]:
                    n = d[0] - mult if d[0] - mult >= mult else int(d[1]/2)
                    df1 = select.query(f"{sX.name} == '{self.classB}' and {y.name} == '{d[f'{y.name}']}'").sample(frac=4/5)
                    df2 = select.query(f"{sX.name} == '{self.classA}' and {y.name} == '{d[f'{y.name}']}'").sample(n=n)
                    df1['swap'], df2['swap'] = False, True
                    df = pd.concat([df1, df2])
                else:
                    n = (mult-d[0])/(d[0]*1.1)
                    df = select.query(f"{sX.name} == '{self.classB}' and {y.name} == '{d[f'{y.name}']}'").sample(frac = n)
                    df['swap'] = False
            i += 1
            p.append(df)
            
        self.oversampled = pd.concat(p).reset_index(drop=True)
    
    
    def generate(self, X, sX, y, **kwargs):
        """
        Generate data by beyond back translation
        X: Series of text to be augmented
        sX: Series variable to be balanced
        y: Series of data target
        verbose: (int), verbosity level, default 1
        swap: (bool) add data to be swapped when balancing, default True
        """
        for obj in (X, sX, y):
            assert isinstance(obj, (pd.Series, np.ndarray)), f"{obj} must be ndarray or Series"
        assert X.shape == sX.shape == y.shape, "args should have same shapes"
        verbose = kwargs.get('verbose', 1)
        swap = kwargs.get('swap', True)
        
        self.oversampling(X, sX, y)
        if verbose: print("Oversampled data is saved in .oversampled")
        
        self.forward = self.translate_(self.oversampled, self.source, self.target)
        if verbose: print("Forward translate data is saved in .forward")
        
        self.generated = self.translate_(self.forward, self.target, self.source)
        if verbose: print("Backward generated data is saved in .generated")
        
        return self.generated



class GenderSwap:
    """
    Swap pronouns, nouns and gendered words
    """
    
    def __init__(self, gender_dict, model):
        """
        gender_dict: (dict) like {'F': 'female', 'M': 'male'}
        model: (str), Word2Vec model
        """
        assert len(gender_dict) == 2, "Only support binary gender"
        self.dict = gender_dict
        self.load_model(model)


    def load_model(self, model):
        if model.endswith(".model"):
            self.model = Word2Vec.load(model)
        else:
            self.model = api.load(model)


    def sentswap(sentence, source, target, thres=.5):
        """
        Swap sentence gender
        sentence: (str) to swap
        source: (str or list) to be substracted
        target: (str or list) to be added
        :return: (str)
        thres: (float) above which swapping is kept
        """
        def swapw2v(model, word, source, target, thres):
            """
            Swap word gender using w2v and similarity
            word: (str) to swap
            *: sentswap args
            :return: (str)
            """
            if not isinstance(source, list): source = [source]
            if not isinstance(target, list): target = [target]
            try:
                sim = model.most_similar(positive=[word, *target],
                                         negative=source,
                                         topn=1)[0]
            except KeyError:
                sim = (word, 1)
            else:
                if word in sim[0] or sim[1] < thres:
                    sim = (word, 1)
            return sim[0]

        sent = " ".join(
            map(lambda x: swapw2v(self.model, x, source, target, thres),
                word_tokenize(sentence)))
        return sent
    

    def generate(self, X, sX, *args, **kwargs):
        """
        Generate swapped sentences
        X: (Series), sentences
        sX: (Series), gender column
        thres: (float), similarity threshold to keep the swapping
        :return: (Series, Series) swapped X, sX
        """
        for obj in (X, sX):
            assert isinstance(obj, pd.Series), f"{obj} must be Series"
        assert X.shape == sX.shape, "args should have same shapes"
        thres = kwargs.get('thres', .5)
        # n_jobs = kwargs.get('n_jobs', 1)
        
        swaped, g_list = list(), list()
        for gender, term in self.dict.items():
            terms = list(self.dict.items())
            terms.remove((gender, term))
            opposite = terms[0][1]
            sentences = X[sX == gender].to_list()
            
            print(f'Swapping for key: {gender}')
            for i, sent in enumerate(sentences):
                print(f'Swapped {i+1} of {len(sentences)}', end='\r', flush=True)
                swaped.append(self.sentswap(
                    sent, opposite, term, thres
                ))
            g_list += [terms[0][0]] * len(sentences)
            
        swaped = pd.Series(swaped, name = X.name)
        g_list = pd.Series(g_list, name = sX.name)
        return swaped, g_list
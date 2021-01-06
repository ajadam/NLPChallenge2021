# -*- coding: utf-8 -*-

__all__ = [
    "BeyondBackTranslator"
]

import pandas as pd
import numpy as np
import boto3
from sklearn.preprocessing import MultiLabelBinarizer

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
    pass
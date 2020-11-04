import sys
sys.path.append('..')
import pandas as pd
from utils import clean_text

df = pd.read_json('../../data/train.json').set_index('Id')

df['description'] = df.loc[:, 'description'].apply(clean_text)

df.to_csv('../../data/trainLemStem.csv')
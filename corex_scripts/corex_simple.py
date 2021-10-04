#%% 
import pandas as pd
import pickle

import os
import numpy as np
import sys
import gensim
import matplotlib.pyplot as plt
sys.path.append(r'C:\Users\aspit\Git\MHDLab-Projects\Energy Storage\nlp_utils')

from nlp_utils import gensim_utils, sklearn_utils, fileio

data_folder = r'C:\Users\aspit\Git\MHDLab-Projects\Energy Storage\data'
db_path = os.path.join(data_folder, 'nlp_search3.db')

df = fileio.load_df(db_path)

# df = df.sample(500, random_state=42)

# text = df['processed_text'].values
texts = df['processed_text'].values

import gensim

texts = [t.split() for t in texts]
bigram = gensim.models.Phrases(texts, threshold=20, min_count=10)
bigram_mod = gensim.models.phrases.Phraser(bigram)

texts_bigram = [bigram_mod[doc] for doc in texts]

texts = [" ".join(t) for t in texts_bigram]

from sklearn.feature_extraction.text import CountVectorizer

# maxx_features = 2**12
vectorizer = CountVectorizer(max_features=None, min_df=2, max_df = 0.9)
X = vectorizer.fit_transform(texts)

feature_names = vectorizer.get_feature_names()
# %%
from corextopic import corextopic as ct
topic_model = ct.Corex(n_hidden=20)  # Define the number of latent (hidden) topics to use.
topic_model.fit(X, words=feature_names, docs=None)
# %%
topics = topic_model.get_topics()
for topic_n,topic in enumerate(topics):
    words,mis = zip(*topic)
    topic_str = str(topic_n+1)+': '+','.join(words[0:5])
    print(topic_str)
# %%
words
# %%

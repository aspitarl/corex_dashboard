#%%
import pandas as pd
import os
import numpy as np
import sys
import gensim
import matplotlib.pyplot as plt
sys.path.append(r'C:\Users\aspit\Git\MHDLab-Projects\Energy Storage\nlp_utils')

# %matplotlib inline

from nlp_utils import  fileio

data_folder = r'C:\Users\aspit\Git\MHDLab-Projects\Energy Storage\data'
db_path = os.path.join(data_folder, 'nlp_search3.db')

df_text = fileio.load_df(db_path)
# df_text = df_text.sample(5000, random_state=42)

texts = df_text['processed_text'].values

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


#%%

from scipy import sparse

sparse.save_npz('data/X.npz', X)


# %%

with open('data/feature_names.txt', 'w', encoding='utf-8') as f:
    f.writelines("%s\n" % feat for feat in feature_names)
# %%


display_text = " <a href=" + df_text['url'] + ">" + df_text['title'] + "</a> logprob=" + df_text['logprob'].apply(str) +" <br>"

# display_text = display_text + "<p>" + df['raw_text'] + "</p>"

df_text['display_text'] = display_text

df_text[['title','display_text','prob']].to_csv('data/metadata.csv')
# %%

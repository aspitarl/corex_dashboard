#%%
import pandas as pd
import os
import numpy as np
import sys
import gensim
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer

input_csv_fp = 'data/input_data.csv'

df_text = pd.read_csv(input_csv_fp, index_col=0)

texts = df_text['processed_text'].values


#Use Gensim to find bigrams
texts = [t.split() for t in texts]
bigram = gensim.models.Phrases(texts, threshold=20, min_count=10)
bigram_mod = gensim.models.phrases.Phraser(bigram)

texts_bigram = [bigram_mod[doc] for doc in texts]

texts = [" ".join(t) for t in texts_bigram]

#Vectorize the resulting texts
vectorizer = CountVectorizer(max_features=None, min_df=2, max_df = 0.9)
X = vectorizer.fit_transform(texts)

feature_names = vectorizer.get_feature_names()


from scipy import sparse

sparse.save_npz('data/X.npz', X)


with open('data/feature_names.txt', 'w', encoding='utf-8') as f:
    f.writelines("%s\n" % feat for feat in feature_names)

if 'url' in df_text:
    display_text = " <a href=" + df_text['url'] + ">" + df_text['title'] + "</a>"
else:
    display_text = df_text['title']
    
if 'prob' in df_text:
    display_text += " logprob=" + df_text['prob'].apply(np.log).apply(lambda x: '%.3f' % x)

display_text += " <br>"

display_text.name = 'display_text'

display_text.to_csv('data/display_text.csv')


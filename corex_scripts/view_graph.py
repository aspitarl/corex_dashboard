#%%

import graphviz
# %%
from graphviz import Source
path = r'C:\Users\aspit\Git\MHDLab-Projects\Energy Storage\topic_modeling\guided_lda\topic-model-example-h\graphs\graph_prune_300.dot'
s = Source.from_file(path)
s.view()
# %%

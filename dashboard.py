#%%
import numpy as np
import pandas as pd
import xarray as xr
import os
import sys

from corextopic import corextopic as ct
import networkx as nx 
from scipy import sparse

from community import community_louvain
import matplotlib.cm as cm

from bokeh.models.sources import ColumnDataSource, CustomJS
from bokeh.io import output_file, show, output_notebook
from bokeh.models import (BoxZoomTool, Circle, HoverTool,
                          MultiLine, Plot, Range1d, ResetTool, TapTool, Div, Circle, MultiLine,
                          Button, TextAreaInput, Slider, Paragraph, TextInput, Spinner)
from bokeh.plotting import figure, show, from_networkx
from bokeh.palettes import Spectral3, Spectral4, Spectral5, Spectral6, Spectral7, Spectral
from bokeh.layouts import column, row, layout
from bokeh.models.widgets import DataTable, DateFormatter, TableColumn, HTMLTemplateFormatter
from bokeh.plotting import figure, show
from bokeh.models.sources import ColumnDataSource
from bokeh.io import curdoc

import _pickle as cPickle

# output_notebook()

BASE_DIR =  os.path.dirname(os.path.abspath(__file__))
data_folder = os.path.join(BASE_DIR, 'data')

X = sparse.load_npz(os.path.join(data_folder, 'X.npz'))

with open(os.path.join(data_folder , 'feature_names.txt'), 'r', encoding='utf-8') as f:
    feature_names = [feat.rstrip() for feat in f.readlines()]

metadata = pd.read_csv(os.path.join(data_folder, 'metadata.csv'), index_col=0)


def get_topic_words(topic_model, num_words=10):

    topic_words = []

    topics = topic_model.get_topics(num_words)
    for topic_n,topic in enumerate(topics):
        words,mis = zip(*topic)
        topic_words.append(", ".join(words))

    return topic_words


from numba import jit

# From nlp_utils, to reduce need for import 
# @jit(nopython=True)
def calc_cov(gamma_di_sub):

    n_docs = gamma_di_sub.shape[0]
    n_topics = gamma_di_sub.shape[1]

    sigma = np.zeros((n_topics,n_topics))
    sigma
    for i in range(n_topics):
        for j in range(n_topics):
            sum = 0
            for doc in range(n_docs):
                sum = sum + gamma_di_sub[doc][i]*gamma_di_sub[doc][j]
            sigma[i][j] = sum

    return sigma


def calc_cov_corex(topic_model, topic_names, doc_names):

    doc_topic_prob = topic_model.p_y_given_x

    n_topics = doc_topic_prob.shape[1]
    n_docs = doc_topic_prob.shape[0]
    
    da_doc_topic = xr.DataArray(doc_topic_prob, coords= {'topic': topic_names, 'doc' : doc_names}, dims = ['doc', 'topic'])

    #Normalize so each topic has total probability one (what does this do in combination with below?)
    theta_ij = da_doc_topic/da_doc_topic.sum('doc')

    #Then normalize so each document has total probability 1
    gamma_di = theta_ij/theta_ij.sum('topic')

    gamma_i = (1/n_docs)*gamma_di.sum('doc')
    gamma_di_sub = gamma_di - gamma_i

    sigma = calc_cov(gamma_di_sub.values)

    da_sigma = xr.DataArray(sigma, coords = {'topic_i': topic_names, 'topic_j': topic_names}, dims = ['topic_i', 'topic_j'])

    return da_sigma, da_doc_topic


def gen_cov_graph(da_sigma, cutoff_weight):

    G = nx.Graph() 

    for topic_i in da_sigma.coords['topic_i'].values:
        G.add_node(topic_i)
        for topic_j in da_sigma.coords['topic_j'].values:
            #xarray puts coordinates in numpy int32, which bokeh doesn't like
            # topic_i = int(topic_i)
            # topic_j = int(topic_j)
            weight = da_sigma.sel(topic_i=topic_i, topic_j=topic_j).item()
            if  weight > cutoff_weight: 
                if topic_i != topic_j:
                    G.add_edge(topic_i,topic_j, weight=np.sqrt(weight))

    for node in G.nodes:
        G.nodes[node]['size'] = 20

    return G

def gen_graph_renderer(G, fill_colors, line_thickness, seed):
    graph_renderer = from_networkx(G, nx.spring_layout, scale=1, seed=seed, center=(0, 0), k=0.05)

    graph_renderer.node_renderer.data_source.add(fill_colors, 'color')
    graph_renderer.node_renderer.data_source.add(line_thickness.values, 'line_thick')    

    graph_renderer.node_renderer.glyph = Circle(radius = 'size', fill_color = 'color', fill_alpha= 1, radius_units='screen', line_color = 'black', line_width='line_thick')
    graph_renderer.edge_renderer.glyph = MultiLine(line_width = 'weight', line_alpha = 0.3)

    return graph_renderer




def get_aligned_docs(da_doc_topic, prob, num=10):
    """Returns docs with max probability for given topic sorted by MA probability. Seems many documents have the same 99.999% topic probability so downselecting with MA""" 

    aligned_docs = []
    for topic in da_doc_topic.coords['topic']:
        s = da_doc_topic.sel(topic=topic).to_series()
        docs = s.sort_values(ascending=False)
        docs = docs.where(docs == docs.iloc[0]).dropna()

        probs = prob[docs.index]
        probs = probs.sort_values(ascending=False)
        top_probs = probs[0:num]

        aligned_docs.append(list(top_probs.index.values))

    aligned_docs = pd.Series(aligned_docs, index = da_doc_topic.coords['topic'])

    return aligned_docs



graph_figure = figure(plot_width=900, plot_height=700,
            x_range=Range1d(-1.1, 1.1), y_range=Range1d(-1.1, 1.1))

node_hover_tool = HoverTool(tooltips=[("index", "@index"), ("text", "@disp_text")])
node_tap_tool = TapTool()
graph_figure.add_tools(node_hover_tool, node_tap_tool)


template="""                
            <div style="background:<%= 
                (function colorfromint(){
                    return(fill_color)
                    }()) %>;"> 
                <%= value %>
            </div>
            """

formatter =  HTMLTemplateFormatter(template=template)

columns = [
        TableColumn(field='partition', title='partition', formatter=formatter, width = 50),
        TableColumn(field='topic', title='Topic', width = 50),
        TableColumn(field='anchor words', title='Anchors', width = 150),
        TableColumn(field='keywords', title='Keywords'),
          ]

data_table = DataTable(source=ColumnDataSource(),
                       columns=columns,
                       width = 800,
                       height=700,
                       fit_columns=False
)


doc = curdoc()

# For displaying updates within functions to front panel
status_paragraph =Div()


#Model controls

generate_model_desc = Paragraph(text='Model Generation: Click to generate a new model. Warning: will take a few minutes, longer for more topics')

text_input = TextAreaInput(rows=8, width= 250, title='Model Anchor Words')

def gen_model_callback(event=None):
    anchor_text = text_input.value
    anchors = anchor_text.splitlines()
        
    #TODO: topics and anchors are misaligned if words not in features
    anchors = [i.split() for i in anchors]
    anchors = [[i[0]] if len(i) == 1 else i for i in anchors ]
    anchor_topics = ['topic_' + str(i) for i in range(len(anchors))]
    s_anchor = pd.Series(anchors, index= anchor_topics, name = 'anchor words')

    missing_words = []
    for topic in s_anchor.index:
        anchors = s_anchor[topic]
        for anchor in anchors:
            if anchor not in feature_names:
                missing_words.append(anchor)
                s_anchor[topic].remove(anchor)
        if len(s_anchor[topic]) == 0:
            s_anchor[topic] = np.nan

    #TODO: make loop above not need to go over index so that the topic index doesn't need to be regenerated.
    s_anchor = s_anchor.dropna()
    anchor_topics = ['topic_' + str(i) for i in range(len(s_anchor))]
    s_anchor.index = anchor_topics

    anchors = s_anchor.values.tolist()

    n_layers = len(anchors) + num_unsup_spinner.value

    topic_model = ct.Corex(n_hidden=n_layers, seed=model_rand_slider.value)  # Define the number of latent (hidden) topics to use.
    topic_model.fit(X, words=feature_names, docs=metadata.index.values, anchors=anchors, anchor_strength=anchor_strength_slider.value) 

    topic_model.anchor_strength = anchor_strength_slider.value

    cPickle.dump(topic_model, open(os.path.join(data_folder, 'topic_model.pkl'), 'wb'))

    s_anchor = s_anchor.apply(" ".join)

    all_topic_names = ['topic_' + str(i) for i in range(n_layers)]
    s_anchor = s_anchor.reindex(all_topic_names)

    s_anchor.to_csv(os.path.join(data_folder, 's_anchor.csv'))
    status_str = 'Done fitting model. '

    if len(missing_words):
        status_str = status_str + 'Could not find anchor words: ' + str(missing_words)

    status_paragraph.text = status_str
    gen_graph_callback()


num_unsup_spinner = Spinner(low=0, high=100, step = 1, value=0, title='number unsupervised topics')
anchor_strength_slider = Slider(start=1, end=10, value=5, title='anchor strength')
model_rand_slider = Slider(start=1, end=100, value = 42, title='model random state')

#need a wrapper to update the text before a long event
#https://stackoverflow.com/questions/59196855/python-bokeh-markup-text-value-cant-update/59199773#59199773
def gen_model_callback_wrapper(event):
    status_paragraph.update(text='Fitting Model, please wait')
    doc.add_next_tick_callback(gen_model_callback)
    
gen_model_button = Button(label = 'Generate Model')
gen_model_button.on_click(gen_model_callback_wrapper)   


button_default_anchors = Button(label='Load Default Anchors')

def load_default_anchors_callback(event):
    with open(os.path.join(data_folder, 'anchor_default.txt')) as f:
        anchor_text = f.read()
    text_input.value = anchor_text

button_default_anchors.on_click(load_default_anchors_callback)


check_word_input = TextInput(title='Check if word in vocabulary')

def check_word_callback(attr, old, new):
    if new in feature_names:
        check_word_input.background = 'green'
    else:
        check_word_input.background = 'red'

check_word_input.on_change('value', check_word_callback)


#Graph controls

generate_graph_desc = Paragraph(text='Graph Generation: Once a topic model is generated, Click Generate Graph to make a graph of connections between topics of model. The edges represent how often topics coocur in a given paper')


def gen_graph_callback(event=None):

    status_paragraph.text = 'Updating Graph'

    if os.path.exists(os.path.join(data_folder, 'topic_model.pkl')):
        topic_model = cPickle.load(open(os.path.join(data_folder, 'topic_model.pkl'), 'rb'))
        s_anchor = pd.read_csv(os.path.join(data_folder, 's_anchor.csv'), index_col=0)['anchor words']

        text_input.value = "\n".join(s_anchor.dropna().values)

        #way to get the number of unsupervised topics. Anchor weight stored in model
        num_unsup_spinner.value = len(s_anchor) - len(s_anchor.dropna())
        anchor_strength_slider.value = topic_model.anchor_strength

        topic_names = s_anchor.index.values
            
        da_sigma, da_doc_topic = calc_cov_corex(topic_model, topic_names, topic_model.docs)

        topic_words = get_topic_words(topic_model, 20)

        topic_keywords = pd.Series(topic_words, index = topic_names, name='topic words')    
        
        G = gen_cov_graph(da_sigma, cutoff_weight_slider.value)

        line_thickness = pd.Series(1, index = topic_names)
        for topic in s_anchor.dropna().index:
            line_thickness[topic] = 5

        line_thickness = line_thickness.reindex(G.nodes)

        for node in G.nodes:
            G.nodes[node]['disp_text'] = topic_keywords[node].replace(',', '\n')

        if len(G.edges) == 0:
            print('did not find any edges in graph')
            # partition = {topic : 0 for topic in topic_names}
            # num_partitions = 1
        # else:
        partition = community_louvain.best_partition(G, resolution = part_res_slider.value, random_state=part_rand_slider.value)
        num_partitions = len(set(partition.values()))
        
        if num_partitions > 7:
            pal = Spectral
        else:
            pal_dict = {1: Spectral3, 2: Spectral3, 3:Spectral3, 4 : Spectral4, 5 : Spectral5,  6: Spectral6, 7:Spectral7}
            pal = pal_dict[num_partitions]
        fill_colors = [pal[i] for i in partition.values()]


        graph_renderer = gen_graph_renderer(G, fill_colors, line_thickness, seed= spring_rand_slider.value)
        graph_figure.renderers = [graph_renderer]

        graph_figure.tools.pop(-1)
        graph_figure.tools.pop(-1)
        graph_figure.add_tools(node_hover_tool, node_tap_tool)

        

        df_table = pd.DataFrame(index= G.nodes)

        topic_part = pd.Series(list(partition.values()), index= G.nodes)
        df_table['partition'] = topic_part


        #Topic names will not be sorted in table as strings...
        df_table['topic'] = [int(topicstr.split('topic_')[1]) for topicstr in G.nodes]

        df_table['fill_color'] = fill_colors

        # df_table = pd.concat([df_table, s_anchor], axis = 1)
        df_table['anchor words'] = s_anchor
        df_table['keywords'] = topic_keywords

        df_table = df_table.sort_values('partition', ascending=False)


        data_table.source.data = df_table

        top_docs = topic_model.get_top_docs()      

        display_texts = metadata['display_text']
        corex_paper_display = []

        for topic_n, topic_docs in enumerate(top_docs):
            docs,probs = zip(*topic_docs)
            docs = "Most Aligned Papers: <br>- " + "- ".join([display_texts[d] for d in docs])
            corex_paper_display.append(docs)
        corex_paper_display = pd.Series(corex_paper_display, index = topic_names)

        MA_paper_display = []
        aligned_docs = get_aligned_docs(da_doc_topic, metadata['prob'])
        for topic in topic_names:
            text = "Papers with highest MA rating for topic: <br>- " + "- ".join([display_texts[d] for d in aligned_docs[topic]])
            MA_paper_display.append(text)

        MA_paper_display = pd.Series(MA_paper_display, index = topic_names)

        def text_callback(attr, old, new):
            if len(new):

                data = graph_renderer.node_renderer.data_source.data
                topic = data['index'][new[0]]
                # disp_text = data['topic_display_texts'][new[0]]

                MA_papers_text.text = MA_paper_display[topic]
                corex_papers_text.text = corex_paper_display[topic]
            else:
                MA_papers_text.text = 'Click on a topic to see papers with highest MA rating for selected topic'
                corex_papers_text.text = 'Click on a topic to see most aligned papers'

        graph_renderer.node_renderer.data_source.selected.on_change("indices", text_callback)
        status_paragraph.text = 'Graph Updated'
    else:
        status_paragraph.text = 'No Topic Model'



gen_graph_button = Button(label = 'Generate Graph')
gen_graph_button.on_click(gen_graph_callback)   

part_res_slider = Slider(start=0.1, end=5.0, value=1.0, step =0.1, title='partition resolution')
cutoff_weight_slider = Slider(start=0, end=5.0, value=0.2, step =0.05, title='edge minimum weight')
part_rand_slider = Slider(start=1, end=100, value = 42, title='partition random state')
spring_rand_slider = Slider(start=1, end=100, value = 42, title='layout random state')

#Paper links

corex_papers_text = Div(style={'overflow-y':'scroll', 'height':'250px', 'width': '800px'})
corex_papers_text.text = 'Click on a topic to see most aligned papers'

MA_papers_text = Div(style={'overflow-y':'scroll', 'height':'250px', 'width': '800px'})
MA_papers_text.text = 'Click on a topic to see papers with highest MA rating for selected topic'


# Layout

model_fitting_sliders = column(num_unsup_spinner, anchor_strength_slider, model_rand_slider)
graph_gen_sliders = column(part_res_slider, part_rand_slider, spring_rand_slider, cutoff_weight_slider)

model_panel = row(column(model_fitting_sliders, check_word_input, ),column(text_input, button_default_anchors), column(generate_model_desc, gen_model_button))
graph_panel = row(graph_gen_sliders, column(generate_graph_desc, gen_graph_button))


layout = column(row(model_panel, graph_panel, status_paragraph), row(graph_figure, data_table), row(corex_papers_text, MA_papers_text))

doc.add_root(layout)

#If there is an existing model already create the graph 
if os.path.exists(os.path.join(data_folder, 'topic_model.pkl')):
    gen_graph_callback(None)
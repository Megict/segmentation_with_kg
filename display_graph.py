from dash import Dash, html, dcc, callback, Output, Input
import plotly.graph_objects as go
import dash_mantine_components as dmc
import networkx as nx
import numpy as np
import json
import pylab
import pandas as pd


app = Dash(__name__)

app.layout = html.Div([

    html.Div([
        html.Button('update', id='do-update'),
        dmc.RangeSlider(value=[0, 200],
                        step = 1,id = 'act-range',style={'width': '25vw', 
                                                    'margin-left' : '2vw' }),
        dcc.Dropdown(id='highlight-node',
                     placeholder="Выберите вершину", style={'width': '20vw', 'margin-left' : '2vw'}),
    ], style={'margin-top' : '2vh', 'display': 'flex'}),

    html.Div([
        dcc.Dropdown(id='center-subgr',
                     placeholder="Выберите вершину", style={'width': '20vw', 'margin-left' : '2vw'}),
        dmc.Slider(value=2,
                        step = 1, min = 1, max = 10, id = 'depth',style={'width': '15vw',  
                                                    'margin-left' : '4vw' }),
        dcc.Dropdown(options = ['links', 'true_edges'], 
                     value = 'links', 
                     id = 'display-mode', 
                     style={'width': '15vw', 'margin-left' : '4vw' }),
        dmc.Slider(value=10,
                   step = 2, min = 0, max = 100, id = 'links',style={'width': '15vw',  
                                                    'margin-left' : '4vw' })
    ], style={'margin-top' : '2vh', 'display': 'flex'}),
    dcc.Graph(id='graph-content', style={
                            'height': '80vh', 'width': '80vw', 'margin-top': '0vh', 'margin-left': '5vw'})
])

from graph_structure import KnowledgeGraph
from graph_analyser import *
from graph_drawer import draw_graph

#=====================
calculate_new_pos = False
calculate_new_links = False
#=====================

my_graph = KnowledgeGraph()
print("loading graph elements...")
my_graph.load("vspu_2019_graph_3_3_links_for_norm_dset")
gran = GraphAnalyser(my_graph)

if calculate_new_pos:
    calculate_new_pos = False

    print("computing layout...")
    my_graph.compute_layout(use_links=False)

if calculate_new_links:
    calculate_new_links = False
    linker = GranLinker(my_graph)
    #link_top_p = 10
    links_f = linker.produce_frequency_links(top_p = 100)
    print("links _f:\t done")
    links_d = linker.produce_distance_links(top_p = 100) # это выбирает из всевозможных пар, а не только существующих связей
    print("links _d:\t done")
    links_s = linker.produce_semantic_links(top_p = 100)
    # print("links _s:\t done")
    gran.imprint_links([links_f, links_d, links_s])
    my_graph.save("test_save")
    print("saved graph")

links_f = gran.present_links(link_type = "freq_link", top_p=10)
print("links _f:\t loaded")
links_d = gran.present_links(link_type = "dist_link", top_p=0.5)
print("links _d:\t loaded")
links_s = gran.present_links(link_type = "sem_link", top_p=0.5)
print("links _s:\t loaded")
display_links = {**links_s, **links_d, **links_f} 
# при отображении пару dthiby связывает только одна ссылка
# print(links_d)

# gran.imprint_links(links)
# my_graph.save("test_save")

@callback(
    Output('highlight-node', 'options'),
    Input('do-update', 'value'))
def set_cities_options(value):
    return list(my_graph.G.nodes())

@callback(
    Output('center-subgr', 'value'),
    Output('center-subgr', 'options'),
    Input('do-update', 'value'))
def set_cities_options(value):
    return list(my_graph.G.nodes)[0], list(my_graph.G.nodes())

@callback(
    Output('graph-content', 'figure'),
    Input('do-update', 'value'),
    Input('act-range', 'value'),
    Input('highlight-node', 'value'),
    Input('center-subgr', 'value'),
    Input('depth', 'value'),
    Input('display-mode', 'value'),
    Input('links', 'value')
)
def update_graph(_, time_range, highlight_node, center_node, depth, edges_display, link_top_p):

    print("construction attempt")
    if edges_display == "links":
        links_ = display_links
        display_edges_ = False
    else:
        links_ = []
        display_edges_ = True

    if center_node is None:
        #center_node = list(my_graph.G.nodes)[0]
        disp_gr = my_graph.G
    else:
        disp_gr = gran.select_subgraph([center_node], depth)

    print("subgraph:\t done")
    pos = nx.kamada_kawai_layout(nx.Graph(disp_gr))
    #pos = my_graph.pos
    print("layout:\t done")

    fig = draw_graph(disp_gr, pos,
                     links = links_,
                     link_color_key = {"freq_link" : "black", "dist_link" : "orange", "sem_link" : "magenta"},
                     display_edges = display_edges_,
                     color_key = "color", 
                     edge_limit_key_name = None, #'locations', 
                     edge_limit_key_values = time_range, 
                     highlight_around= [highlight_node] if highlight_node != None else [])
    print("done")
    return fig

if __name__ == '__main__':
    app.run(debug=True)
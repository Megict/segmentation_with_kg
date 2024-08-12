from dash import Dash, html, dcc, callback, Output, Input, State
import plotly.graph_objects as go
import dash_mantine_components as dmc

import networkx as nx
import numpy as np
import json
import pylab
import pandas as pd


app = Dash(__name__)


# with open("graphs_", 'r') as f:
#     graphs = json.load(f)["g"]
# sentence_graphs = [nx.node_link_graph(g) for g in graphs]

app.layout = html.Div([

    html.Div([
        html.Button('update', id='do-update', style={'width': '5vw', 'margin-left' : '2vw'}),
        #dcc.Input(id="text-input", placeholder="input sentence", style={'width': '60vw', 'height': '3vh', 'margin-left' : '5vw', 'margin-up' : '5vh'})
        dcc.Input(id='itext_input', placeholder="Введите текст",  style={'width': '20vw', 'margin-left' : '5vw', 'margin-up' : '5vh'}),

    ], style={'margin-top' : '2vh', 'display': 'flex'}),

    dcc.Graph(id='graph-content', style={
                            'height': '80vh', 'width': '80vw', 'margin-top': '0vh', 'margin-left': '5vw'})
])

from graph_builder import GraphBuilder
from graph_drawer import draw_graph

my_builder = GraphBuilder()
# my_builder.init_parser()


@callback(
    Output('graph-content', 'figure'),
    Input('do-update', 'value'),
    Input('itext_input', 'value')
)
def update_graph(update_req_cnt, text):
    print("============================")
    print(update_req_cnt)
    if text is None:
        text = 0
        #text = "Построение грубых сеток заключается в удалении из каждой четверки последовательных узлов двух средних точек как показано на рис"
    print(text)

    print("construction attempt")
    #graph = my_builder.extract_noun_phrases(text, clear_hanging_nodes=True, add_focus=["среда"])
    graph = my_builder.create_syntax_graph_for_sentence(text)
    #display_graph = my_builder.filter_graph_(graph)
    print(graph.nodes())

    print("subgraph:\t done")
    pos = nx.kamada_kawai_layout(nx.Graph(graph))
    #pos = my_graph.pos
    print("layout:\t done")

    fig = draw_graph(graph, pos,
                     link_color_key = {"freq_link" : "black", "dist_link" : "orange", "sem_link" : "magenta"},
                     display_edges = True,
                     color_key = "color", 
                     edge_limit_key_name = None
                     )
    print("done")
    return fig

if __name__ == '__main__':
    app.run(debug=True)
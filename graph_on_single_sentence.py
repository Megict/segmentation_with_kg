from dash import Dash, html, dcc, callback, Output, Input
import plotly.graph_objects as go
import dash_mantine_components as dmc
import networkx as nx
import numpy as np
import json
import pylab
import pandas as pd
from copy  import deepcopy as cp

from graph_drawer import draw_graph
from graph_builder import GraphBuilder

builder = GraphBuilder()

def find_name_group(graph, starting_point, ng = set()):
    graph = nx.Graph(graph)
    if graph.nodes[starting_point]["pos"] == "VERB":
        return ng
    else:
        ng.update({starting_point})
    for n in graph.neighbors(starting_point):
        if graph.nodes[n]["pos"] != "VERB" and n not in ng:
            ng.update({n})
            ng = find_name_group(graph, n, ng)
    return ng        
        

sentence = "В данном примере мы рассматриваем способ построения синтаксичекого графа на основе одного предложения"

while True:
     print(sentence)
     sentence_graph = builder.create_syntax_graph_for_sentence(sentence)
     subgraphs = []

     ng_graph = nx.Graph()
    
     for node in sentence_graph:
         if sentence_graph.nodes[node]['pos'] != "VERB":
             ng = find_name_group(sentence_graph, node, ng = set())

             # ищем корень именной группы
            
             root = "none"
             for node in ng:
                 sucs = [n for n in sentence_graph.successors(node) if n in ng]
                 print(node, len(sucs))
                 if len (sucs) == 0:
                     root = node

             ng_graph.add_node(' '.join([elm for elm in sorted(ng)]), pos = "NG", part = cp(ng), root = root)


     for edge in sentence_graph.edges:
         if sentence_graph.nodes[edge[1]]['pos'] != "VERB" and sentence_graph.nodes[edge[0]]['pos'] != "VERB":
             continue

         if sentence_graph.nodes[edge[0]]['pos'] != "VERB" and sentence_graph.nodes[edge[1]]['pos'] == "VERB":

             for n in ng_graph.nodes:
                 if edge[0] in ng_graph.nodes[n]['part']:
                     ng_graph.add_node(edge[1], pos = 'VERB', part = set([edge[1]]))
                     ng_graph.add_edge(n, edge[1])
                     break

         if sentence_graph.nodes[edge[0]]['pos'] == "VERB" and sentence_graph.nodes[edge[1]]['pos'] != "VERB":
            
             for n in ng_graph.nodes:
                 if edge[1] in ng_graph.nodes[n]['part']:
                     ng_graph.add_node(edge[0], pos = 'VERB', part = set([edge[0]]))
                     ng_graph.add_edge(edge[0], n)
                     break

         if sentence_graph.nodes[edge[0]]['pos'] == "VERB" and sentence_graph.nodes[edge[1]]['pos'] == "VERB": 
             # по идее так бывает только с деепричастиями, которые определяются как глаголы
             ng_graph.add_node(edge[0], pos = 'VERB', part = set([edge[0]]))
             ng_graph.add_node(edge[1], pos = 'VERB', part = set([edge[1]]))
             ng_graph.add_edge(edge[0], edge[1])

     # убираем мертвые ссылки
     # for node in cp(ng_graph.nodes):
     #     if len([n for n in ng_graph.neighbors(node)]) <= 1 and ng_graph.nodes[node]['pos'] == "VERB":
     #         ng_graph.remove_node(node)
            
     for elm in ng_graph.nodes:
         if ng_graph.nodes[elm]["pos"] == "NG":
             subgraphs.append(sentence_graph.subgraph(ng_graph.nodes[elm]['part']))

     print("\n\n=================")
     for sg in subgraphs:
         for node in sg.nodes:
             print(node)
         print("-------\/-------")
         for edge in sg.edges:
             print(edge)
         print("=================")
            

     print(" ------------------------------------------------------------------- ")
     for node in ng_graph.nodes:
         print(f"{node} \t|\t==\t|\t {ng_graph.nodes[node]}")
     print(" ------------------------------------------------------------------- ")
     for edge in ng_graph.edges:
         print(f"{edge} \t|\t==\t|\t {ng_graph.edges[edge]}")

        
     sentence = input()

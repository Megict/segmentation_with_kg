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

from graph_on_single_alt import ng_graph

file_name = input()
f = open(file_name)

ng_s = []
for line in f:
    ng_ = ng_graph(line)
    ng_s.append([])
    ng_s[-1] = [ng_.nodes[n]["ng_graph"] for n in ng_.nodes if ng_.nodes[n]["pos"] == "NG"]

for elm in ng_s:
    print(elm)
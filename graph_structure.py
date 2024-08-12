import networkx as nx
import numpy as np
import pylab
import os

from copy import deepcopy as cp

import json

from utils import remap_keys, remap_back

class KnowledgeGraph:

    def __init__(self):
        self.pos = None
        self.G = nx.DiGraph()
        self.links = {}  # ссылки хранятся в виде отдельного словаря

        self.document_base = {} # словарь, где хранятся имена файлов и длины текстов в них
                                # все файлы должны называться по-разному
    
    # ----------------------------------------------- слияние графов -----------------------------------------------
    # это основной и единственный метод добавления новых узлов
        
    def __merge_graph__(self, merge_Graph): # используется только для мерджа графа одного текста как часть метода parse_file_and_fill_graph
        # сделать так, чтобы при слиянии новые вершины помечались
        for node_to_add in merge_Graph.nodes:
            if node_to_add in self.G.nodes:
                # здесь слияние аттрибутов
                self.G.nodes[node_to_add]['locations'] =\
                    self.G.nodes[node_to_add]['locations'] +\
                    merge_Graph.nodes[node_to_add]['locations']
            else:
                self.G.add_node(node_to_add, **merge_Graph.nodes[node_to_add])

        for edge_to_add in merge_Graph.edges:
            if edge_to_add in self.G.edges:
                self.G.edges[edge_to_add]["label"][0] = self.G.edges[edge_to_add]["label"][0] + merge_Graph.edges[edge_to_add]["label"][0]
                self.G.edges[edge_to_add]['locations'] =\
                    self.G.edges[edge_to_add]['locations'] +\
                    merge_Graph.edges[edge_to_add]['locations']
            else:
                self.G.add_edge(*edge_to_add, **merge_Graph.edges[edge_to_add])

    def absorb_other(self, other):
        # при слиянии с другим графом, нужно также объеденить базы текстов
        for name in other.document_base:
            self.document_base[name] = other.document_base[name]
        # сделать так, чтобы при слиянии новые вершины помечались
        for node_to_add in other.G.nodes:
            if node_to_add in self.G.nodes:
                # здесь слияние аттрибутов
                self.G.nodes[node_to_add]['locations'] =\
                    self.G.nodes[node_to_add]['locations'] +\
                    other.G.nodes[node_to_add]['locations']
            else:
                self.G.add_node(node_to_add, **other.G.nodes[node_to_add])

        for edge_to_add in other.G.edges:
            if edge_to_add in self.G.edges:
                self.G.edges[edge_to_add]["label"][0] = self.G.edges[edge_to_add]["label"][0] + other.G.edges[edge_to_add]["label"][0]
                self.G.edges[edge_to_add]['locations'] =\
                    self.G.edges[edge_to_add]['locations'] +\
                    other.G.edges[edge_to_add]['locations']
            else:
                self.G.add_edge(*edge_to_add, **other.G.edges[edge_to_add])
        
        del other

    def remove_node(self, node_to_remove, remove_links = False):
        # удаляются все ребра, связанные с этим узлом и сам узел
        # ссылки остаются, они все равно обновятся при следующем пересчете
        self.G.remove_node(node_to_remove)
        if remove_links == True:
            for l_type in self.links:
                for l in cp(self.links[l_type]):
                    if node_to_remove in l:
                        self.links[l_type].pop(l)
        
    def remove_edge(self, edge_to_remove, remove_links = False):
        # удаляется ребро
        self.G.remove_edge(*edge_to_remove)

    # ------------------------------------- сохранение и загрузка --------------------------------------------
    # сохранение и загрузка всего графа целиком
        
    def save(self, path_to_folder):

        json_friendly_layout = {n : list(npos) for n,npos in self.pos.items()}
        json_friendly_links = cp(self.links)

        for link_type in json_friendly_links.keys():
            for link in json_friendly_links[link_type].keys():
                json_friendly_links[link_type][link]["score"] = str(json_friendly_links[link_type][link]["score"])
            json_friendly_links[link_type] = remap_keys(json_friendly_links[link_type])

        try:
            os.mkdir(path_to_folder)
        except OSError:
            pass

        with open(path_to_folder + "/" + "graph_structure.kg", 'w') as f:
            json.dump({"g" : nx.node_link_data(self.G), 
                       "db" : self.document_base}, f)
        with open(path_to_folder + "/" + "links.kg", 'w') as f:
            json.dump({"l" : json_friendly_links},f)
        with open(path_to_folder + "/" + "layout.kg", 'w') as f:
            json.dump({"p" : json_friendly_layout},f)
                       

    def load(self, path_to_folder):
        with open(path_to_folder + "/" + "graph_structure.kg", 'r') as f:
            file = json.load(f)
            self.G = nx.node_link_graph(file["g"])
            self.document_base = file["db"]

        try:
            with open(path_to_folder + "/" + "links.kg", 'r') as f:
                file = json.load(f)
                links = file["l"]
                for link_type in links.keys():
                    links[link_type] = remap_back(links[link_type])
                    for link in links[link_type].keys():
                        links[link_type][link]["score"] = float(links[link_type][link]["score"])
                self.links = links
        except OSError:
            print("failed to load links")

        try:
            with open(path_to_folder + "/" + "layout.kg", 'r') as f:
                file = json.load(f)
                self.pos = file["p"]
        except OSError:
            print("failed to load layout")

    # сохранение и загрузка элементов графа по отдельности

    def present_links(self, link_type = None):
        # все ссылки вернуть нельзя, потому что будут пересечения
        return self.links[link_type]
    
    def present_link_score(self, link_type : str, link_pair : tuple):
        try:
            link_score = self.links[link_type][link_pair]["score"]
        except KeyError:
            # может есть обратная ссылка
            try:
                link_score = self.links[link_type][(link_pair[1], link_pair[0])]["score"]
            except KeyError:
                # такой ссылки вообще нет
                return None
        return link_score
        

    # --------------------------------------- средства простого отображения -----------------------------------------------------------    

    def compute_layout(self, use_links = False):
        # надо придумать, как учитывать тоьлко ссылки с высокими score
        # идея в том, что граф самодостаточен и его можно отобразить без прочих модулей
        # поэтому layout считается в нем
        layout_graph = nx.Graph(self.G)
        if use_links:
            for link_type in self.links.items():
                for link in link_type[1]:
                    layout_graph.add_edge(*link)
        self.pos = nx.kamada_kawai_layout(layout_graph)
    
    def display_pylab(self, display_type, center = None, depth = None, layout = True):
        # хранить цвета, edge properties, edge labrls etc в самом графе
        if display_type == "components":
            for c in nx.weakly_connected_components(self.G):
                if len(c) > 2:
                    sg = self.G.subgraph(c)
                    colors = [sg.nodes()[n]["color"] for n in sg.nodes()]
                    pylab.figure(figsize=(10,8))
                    nx.draw(sg, pos = self.pos, with_labels = True, node_color = colors)
                    if self.pos != None:
                        nx.draw_networkx_edge_labels(sg, pos = self.pos, edge_labels = 
                                                        {e : sg.edges()[e]['label'][1] +\
                                                        ' ' + str(sg.edges()[e]['locations'][0][1]) + ' ' +\
                                                        '(' + str(sg.edges()[e]['label'][0]) + ')'
                                                        for e in sg.edges()})
                    pylab.show()

        if display_type == "full":
            colors = [self.G.nodes()[n]["color"] for n in self.G.nodes()]
            pylab.figure(figsize=(10,8))
            nx.draw(self.G, pos = self.pos, with_labels = True, node_color = colors)
            if self.pos != None:
                nx.draw_networkx_edge_labels(self.G, pos = self.pos, edge_labels = 
                                                {e : self.G.edges()[e]['label'][1] +\
                                                ' ' + str(self.G.edges()[e]['locations'][0][1]) + ' ' +\
                                                '(' + str(self.G.edges()[e]['label'][0]) + ')'
                                                for e in self.G.edges()})
            pylab.show()

        if display_type == "subgraph" and center is not None:
            sg, pos = self.pull_subgraph(center, depth = 1 if depth is None else depth, layout = True)
            if not layout:
                pos = None
            colors = [sg.nodes()[n]["color"] for n in sg.nodes()]
            pylab.figure(figsize=(10,8))
            nx.draw(sg, pos = pos, with_labels = True, node_color = colors)
            if pos != None:
                nx.draw_networkx_edge_labels(sg, pos = pos, edge_labels = 
                                                {e : sg.edges()[e]['label'][1] +\
                                                ' ' + str(sg.edges()[e]['locations'][0][1]) + ' ' +\
                                                '(' + str(sg.edges()[e]['label'][0]) + ')'
                                                for e in sg.edges()})
            pylab.show()

    def pull_subgraph(self, center, depth, layout = False):
        # выдать часть графа, состоящую из узлов, находящихся на расстоянии depth и меньше от центра
        subgraph_nodes = [center]
        cur_frontier = [center]
        for _ in range(depth):
            new_fronteer = []
            for fnd in cur_frontier:
                subgraph_nodes += self.G.neighbors(fnd)
                subgraph_nodes += self.G.predecessors(fnd)
                new_fronteer += self.G.neighbors(fnd)
                new_fronteer += self.G.predecessors(fnd)
            cur_frontier = cp(new_fronteer)
        
        sg = self.G.subgraph(subgraph_nodes)

        if layout:
            return sg, nx.kamada_kawai_layout(sg)
        else:
            return sg
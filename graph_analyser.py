import networkx as nx
from tqdm import tqdm
import numpy as np
from copy import deepcopy as cp

import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from graph_structure import KnowledgeGraph

# Gran naming conventions
# =======================
# Take something that is given and write it to structure        imprint
# Take something that already exisits and RETURN it             present
# Calculate something and RETURN it (aux private method)        calculate
# Calculate something and RETURN it (public method)             produce
#

class GraphAnalyser():

    def __init__(self, graph_structure_ : KnowledgeGraph):
        self.graph_structure = graph_structure_
        self.selected_subgraph = None
        self.lm = None
    
    def select_subgraph(self, root_nodes, # узлы, которые надо включить в подграф
                            radius, # узлы, до которых можно дойти radius шагами от корневых узлов, также включаюься
                        ):
        
        subgraph_nodes = cp(root_nodes)
        cur_frontier = cp(root_nodes)
        for _ in range(radius):
            new_fronteer = []
            for fnd in cur_frontier:
                subgraph_nodes += self.graph_structure.G.neighbors(fnd)
                subgraph_nodes += self.graph_structure.G.predecessors(fnd)
                new_fronteer += self.graph_structure.G.neighbors(fnd)
                new_fronteer += self.graph_structure.G.predecessors(fnd)
            cur_frontier = cp(new_fronteer)

        self.selected_subgraph = self.graph_structure.G.subgraph(subgraph_nodes)
        
        return self.selected_subgraph

    def init_language_model(self, model_ = "mlsa-iai-msu-lab/sci-rus-tiny"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SentenceTransformer(model_)
        model = model.eval()
        model = model.to(self.device)
        self.lm = model
        
    def clear_node_list(self, node_list):
        # вернуть список, в котором есть только те узлы, что есть в графе
        clear_node_list = []
        for node in node_list:
            try:
                _ = self.graph_structure.G.nodes[node]
                clear_node_list.append(node)
            except KeyError:
                continue
        return clear_node_list
    
    # запись / чтение ссылок 

    def imprint_links(self, links : list):
        # записать ссылки в структуру графа
        # на вход идет список словарей с ссылкам. в каждом словаре ссылки одного типа
        for link_batch in links:
            link_batch_type = list(link_batch.items())[0][1]["type"]
            
            if link_batch_type in self.graph_structure.links.keys():
                for link in link_batch.items():
                    self.graph_structure.links[link_batch_type][link[0]] = link[1]
            else:
                self.graph_structure.links[link_batch_type] = link_batch

    def present_links(self, link_type = None, top_p = 100):
        # если надо отобразить ссылки, берем их отсюда
        if link_type is None and top_p != 100:
            print("you must specify link type to request cutoff")
        links = self.graph_structure.present_links(link_type)
        # здесь выполняется сортировка ссылок, чтобы их score попадал в top_p
        if top_p != 100:
            link_number = int(len(links.items()) * top_p / 100)
            link_values = dict(list(sorted(links.items(), key = lambda x: x[1]["score"], reverse = False))[:link_number])
        else:
            link_values = links
        return link_values
    
    # вспомогательные функции

    def calculate_cosine_difference(self, lhs, rhs):
        if self.lm is None:
            self.init_language_model()
        model = self.lm

        texts = [lhs, rhs]
        with torch.no_grad():
            embed_1, embed_2 = model.encode(texts)
        return max(1 - cosine_similarity([embed_1], [embed_2])[0][0], 0)
        # идея в том, что чем меньше значение, тем ближе слова
        # странно, что одно и то же слово не дает 1
    
    def calculate_text_distance(self, lhs_node, rhs_node, mode = 'shortest'):
        # возвращает расстояние между вершинами графа в тексте (если они встречались в одном тексте) в символах
        # может возвращать наименьшее расстояние или среднее
            # shortest  самое короткое расстояние
            # avg       среднее расстояние между совпадениями
            # avg_all   среднее расстояние между всеми встречами, если в разных текстах, то считается как 1
        if mode == 'shortest':
            dist = None
            for loc in self.graph_structure.G.nodes[lhs_node]['locations']:
                for other_loc in self.graph_structure.G.nodes[rhs_node]['locations']:
                    if loc['t'] == other_loc['t']:
                        
                        if dist is None or abs(loc['p'] - other_loc['p']) < dist:
                            dist = abs(loc['p'] - other_loc['p']) / self.graph_structure.document_base[loc['t']]['len']
            if dist == None:
                dist = 1

        if mode[0:3] == 'avg':
            dist = 0
            match_cnt = 0
            for loc in self.graph_structure.G.nodes[lhs_node]['locations']:
                for other_loc in self.graph_structure.G.nodes[rhs_node]['locations']:
                    if loc['t'] == other_loc['t']:
                        dist += abs(loc['p'] - other_loc['p']) / self.graph_structure.document_base[loc['t']]['len']
                        match_cnt += 1
                    else:
                        if mode == 'avg_all':
                            dist += 1
                            match_cnt += 1

            if match_cnt != 0:
                dist /= match_cnt

        return dist
    
    # возварт определенных значений, которые могут храниться в ссылках
    
    def produce_distance_score(self, lhs_node : str, rhs_node : str, mode = "avg", verbouse = False):
        dist_score = self.graph_structure.present_link_score("dist_link", (lhs_node, rhs_node))
        if dist_score is None:
            if verbouse:
                print("no dist link, recalculating")
            dist_score = self.calculate_text_distance(lhs_node, rhs_node, mode=mode)
            self.imprint_links([{(lhs_node, rhs_node) : {"type" : "dist_link", "score" : dist_score}}])
        return dist_score

    def produce_semantic_score(self, lhs_node : str, rhs_node : str, verbouse = False):
        sem_score = self.graph_structure.present_link_score("sem_link", (lhs_node, rhs_node))
        if sem_score is None:
            if verbouse:
                print("no sem link, recalculating")
            sem_score = self.calculate_cosine_difference(lhs_node, rhs_node)
            self.imprint_links([{(lhs_node, rhs_node) : {"type" : "sem_link", "score" : sem_score}}])
        return sem_score

class GranLinker(GraphAnalyser):
    # это структура для работы со ссылками
    # ------------------------------- работа со ссылками --------------------------------------------------
    # вспомогательные функции
    def __init__(self, graph_structure_):
        super().__init__(graph_structure_)
    
    # основные функции создания ссылок
    def __calculate_cosine_differences(self, texts):
        # найти схожести всех текстов в массиве со всеми
        if self.lm is None:
            self.init_language_model()
        model = self.lm

        with torch.no_grad():
            embeds = model.encode(texts)
        # print(embeds)
        # sim = cosine_similarity(embeds)
        # print(sim)
        # print(len(embeds))
        # print(len(sim))
        sim = []
        sim_matr = cosine_similarity(embeds)

        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                sim.append((texts[i], texts[j], max(1 - sim_matr[i][j],0)))
                
        return sim
        
    def __calculate_frequency_scores(self,
                                   for_entire_graph = False):
        if self.selected_subgraph is None:
            for_entire_graph = True
        graph = self.graph_structure.G if for_entire_graph else self.selected_subgraph

        edge_freq = []
        for edge in graph.edges:
            edge_freq.append(len(graph.edges[edge]["locations"]))
        #fmax = max(edge_freq)
        for i in range(len(edge_freq)):
            edge_freq[i] = 1 / edge_freq[i]
        return edge_freq
        
    def produce_frequency_links(self,
                                top_p = 10, # число от 1 до 100, в результат включаются n% граней с наибольшим значением веса
                                for_entire_graph = False):
        edge_freq = self.__calculate_frequency_scores(for_entire_graph)

        if self.selected_subgraph is None:
            for_entire_graph = True
        graph = self.graph_structure.G if for_entire_graph else self.selected_subgraph

        # осталось отсеить грани с наибольшими весами
        # вместо вектора весов вернем новые грани
        link_number = int(len(edge_freq) * top_p / 100)
        link_ids = dict(list(sorted({lnum : lscore for lnum, lscore in enumerate(edge_freq)}.items(), key = lambda x: x[1], reverse = False))[:link_number])
        links = {}
        for edge_n, edge in enumerate(graph.edges):
            if edge_n in link_ids.keys():
                links[edge] = {}
                links[edge]["type"] = "freq_link"
                links[edge]["score"] = link_ids[edge_n]

        return links

    def produce_distance_links(self,
                               top_p = 10, # число от 1 до 100, в результат включаются n% граней с наименьшим значением расстояний
                               for_entire_graph = False):
        if self.selected_subgraph is None:
            for_entire_graph = True
        graph = self.graph_structure.G if for_entire_graph else self.selected_subgraph

        n_list = list(graph.nodes)
        dist_values = []
        for lhs_p in tqdm(range(len(graph.nodes))):
            for rhs_p in range(lhs_p, len(graph.nodes)):
                if lhs_p != rhs_p:
                    dist_values.append(((n_list[lhs_p], n_list[rhs_p]), self.calculate_text_distance(n_list[lhs_p], n_list[rhs_p], mode="avg")))
                    
        link_number = int(len(dist_values) * top_p / 100)
        link_values = dict(list(sorted(dist_values, key = lambda x: x[1], reverse = False))[:link_number])

        links = {}
        for link, score in link_values.items():
            links[link] = {}
            links[link]["type"] = "dist_link"
            links[link]["score"] = score
        return links
    
    def produce_semantic_links(self,
                               top_p = 10, # число от 1 до 100, в результат включаются n% граней с наименьшим значением расстояний
                               for_entire_graph = False):
        if self.selected_subgraph is None:
            for_entire_graph = True
        graph = self.graph_structure.G if for_entire_graph else self.selected_subgraph

        sim_values = self.__calculate_cosine_differences(list(graph.nodes))
                    
        link_number = int(len(sim_values) * top_p / 100)
        link_values = dict(list(sorted(sim_values, key = lambda x: x[1], reverse = False))[:link_number])

        links = {}
        for link, score in link_values.items():
            links[link] = {}
            links[link]["type"] = "sem_link"
            links[link]["score"] = score
        return links


from networkx.exception import NetworkXNoPath
from networkx.classes.function import path_weight
import yargy
from yargy.pipelines import morph_pipeline
from utils import shanon_entropy
from utils import intersection
from utils import union
from utils import sigmoid

class GranGapEstimater(GraphAnalyser):
    # это структура для поиска расстояний мажду двумя подграфами
    def __init__(self, graph_structure_: KnowledgeGraph):
        super().__init__(graph_structure_)
        self.dist_graph = None
        try:
            _ = self.graph_structure.links["gd_link"]
        except KeyError:
            self.graph_structure.links["gd_link"] = {}

        self.subgraph_inflation = 0 # значения для параметров счета расстояний по умолчанию
        self.use_weight_for_pathing = False
        self.trace = False

        self.tokenizer = yargy.tokenizer.MorphTokenizer()
        nodes_norm = [" ".join([x.value for x in self.tokenizer(item)])
            for item in self.graph_structure.G.nodes]
        self.nodes_norm = nodes_norm
        self.nodes_parser = yargy.Parser(morph_pipeline(nodes_norm))



    def calculate_distances_to_all_nodes(self, start):
        # нужно использовать bfs и добавлять ссылки
        return None

    # поиск взвешенных расстояний по ссылкам / графу
    def calculate_link_distance(self, lhs_node : str, rhs_node : str,
                                dist_weight_weight = 1, # значение от 0 до 2
                                sem_weight_weight = 1, # значение от 0 до 2
                                entropy_weight = 0.5 # значение от 0 до 2
                                
                                ):
        # найти вес ссылки между вершинами графа, нужны полносвязные ссылки
        
        dist_weight = self.produce_distance_score(lhs_node, rhs_node, verbouse=False)
        sem_weight = self.produce_semantic_score(lhs_node, rhs_node, verbouse=False)

        link_distanse = dist_weight_weight*dist_weight + sem_weight_weight*sem_weight + entropy_weight*(1 / shanon_entropy(lhs_node) + 1 / shanon_entropy(rhs_node)) # делим на 3 чтобы нормализовать все к 1
        # чем расстояние короче, тем ближе слова
        return sigmoid(link_distanse, c = 1) - 0.5

    def __create_dist_graph(self):
        self.dist_graph = nx.Graph()
        self.dist_graph.add_nodes_from(self.graph_structure.G.nodes())
        print("created distance graph")
        for link in self.graph_structure.links["freq_link"]:
            self.dist_graph.add_edge(*link, **self.graph_structure.links["freq_link"][link])
        
    def produce_graph_distance_score(self, lhs_node : str, rhs_node : str, verbouse = False):
        # расстояние по графу (кэшируется в ссылках)
        dist_score = self.graph_structure.present_link_score("gd_link", (lhs_node, rhs_node))
        if dist_score is None:
            if verbouse:
                print("no dist link, recalculating")
            shortest_graph_path = nx.shortest_path(self.dist_graph, lhs_node, rhs_node) # кратчайший путь, но не путь с наим. весом
            shortest_graph_distanse = path_weight(self.dist_graph, shortest_graph_path, weight = "score")
            dist_score = shortest_graph_distanse
            self.imprint_links([{(lhs_node, rhs_node) : {"type" : "gd_link", "score" : dist_score}}])
        return dist_score
    
    def calculate_graph_distance(self, lhs_node : str, rhs_node : str, 
                                 no_path_weight = 0.5): # что вернуть, если пути между двумя вершинами вовсе нету
        # найти длину пути по синтаксическому графу
        # в качестве весов используются частотные ссылки
        # чем меньше значение ссылки, тем чаще встречается это ребро
        if self.dist_graph is None:
            self.__create_dist_graph()

        try:
            shortest_graph_distanse = sigmoid(self.produce_graph_distance_score(lhs_node, rhs_node), c = 0.5) - 0.5
        except NetworkXNoPath:
            shortest_graph_distanse = no_path_weight

        # тут считается самый короткий путь по невзвешенному графу, а затем считается его вес
        # это может быть бессмысленно
        return shortest_graph_distanse
    
    # def calculate_graph_weight(self, lhs_node : str, rhs_node : str,
    #                            no_path_weight = 0.5):# что вернуть, если пути между двумя вершинами вовсе нету
    #     # работает долго и разницы никакой но вроде "правильнее"
    #     # найти длину пути по синтаксическому графу
    #     # в качестве весов используются частотные ссылки
    #     # чем меньше значение ссылки, тем чаще встречается это ребро
    #     if self.dist_graph is None:
    #         self.__create_dist_graph()

    #     try:
    #         shortest_graph_weight = sigmoid(nx.shortest_path_length(self.dist_graph, lhs_node, rhs_node, weight = "score")) - 0.5 # кратчайший путь
    #     except NetworkXNoPath:
    #         shortest_graph_weight = no_path_weight

    #     # тут считается самый короткий путь по невзвешенному графу, а затем считается его вес
    #     # это может быть бессмысленно
    #     return shortest_graph_weight
    
    def estimate_gap_simple(self, lhs_text, rhs_text, trace = False, 
                                                      subgraph_inflation = None,
                                                      node_detection_method = 'native',
                                                      zero_overlay_result = 0):
        
        subgraph_inflation_ = subgraph_inflation if subgraph_inflation else self.subgraph_inflation
        trace_ = trace if trace else self.trace
        
        if node_detection_method == 'native':
        # старый способ извлечения вершин из текста
            lhs_words = lhs_text.split()
            rhs_words = rhs_text.split()

            lhs_core_graph_nodes = self.clear_node_list(lhs_words)
            rhs_core_graph_nodes = self.clear_node_list(rhs_words)
        else:
            lhs_core_graph_nodes = []
            for m in self.nodes_parser.findall(lhs_text):
                tokens_norm = [x.normalized for x in m.tokens]
                lhs_core_graph_nodes.append(" ".join(tokens_norm))
                
            rhs_core_graph_nodes = []
            for m in self.nodes_parser.findall(rhs_text):
                tokens_norm = [x.normalized for x in m.tokens]
                rhs_core_graph_nodes.append(" ".join(tokens_norm))

        lhs_graph = self.select_subgraph(lhs_core_graph_nodes, radius = subgraph_inflation_)
        rhs_graph = self.select_subgraph(rhs_core_graph_nodes, radius = subgraph_inflation_)

        lhs_graph_nodes = lhs_graph.nodes
        rhs_graph_nodes = rhs_graph.nodes

        if trace_:
            print(f"lens: \t {len(lhs_graph.nodes.keys())} {len(rhs_graph.nodes.keys())}")

        intersection_ = intersection(lhs_graph_nodes, rhs_graph_nodes)
        union_ = union(lhs_graph_nodes, rhs_graph_nodes)
        
        if trace_:
            print(f"i: \t {len(intersection_)}")
            print(f"u: \t {len(union_)}")

        if len(union_) == 0:
            return zero_overlay_result

        return 1 - (len(intersection_) / len(union_)) # от 0 до 1
    
    
    def estimate_gap_graph_distance(self, lhs_text, rhs_text, 
                                       trace = False, 
                                       subgraph_inflation = None, # до какой глубины соседние вершины будут добавляться в подграф
                                       use_weight_for_pathing = None, # гораздо медленнее, но в теории лучше

                                       graph_distance_weight = 1, # значение от 0 до 2, на него домножается расстояние по графу

                                       graph_distance_finder_params = {},

                                       zero_overlay_result = 0,
                                       ): 
        use_weight_for_pathing_ = use_weight_for_pathing if use_weight_for_pathing else self.use_weight_for_pathing
        subgraph_inflation_ = subgraph_inflation if subgraph_inflation else self.subgraph_inflation
        trace_ = trace if trace else self.trace
        
        lhs_words = lhs_text.split()
        rhs_words = rhs_text.split()

        lhs_core_graph_nodes = self.clear_node_list(lhs_words)
        rhs_core_graph_nodes = self.clear_node_list(rhs_words)

        lhs_graph = self.select_subgraph(lhs_core_graph_nodes, radius = subgraph_inflation_)
        rhs_graph = self.select_subgraph(rhs_core_graph_nodes, radius = subgraph_inflation_)

        lhs_graph_nodes = lhs_graph.nodes
        rhs_graph_nodes = rhs_graph.nodes

        if trace_:
            print(f"lens: \t {len(lhs_graph.nodes.keys())} {len(rhs_graph.nodes.keys())}")

        # меряем расстояния от каждогй вершины графа 1 до ближайшей вершины графа 2
        distances = []
        for source_node in lhs_graph_nodes:
            min_dist = None
            for target_node in rhs_graph_nodes:
                graph_distance = self.calculate_graph_weight(source_node, target_node, **graph_distance_finder_params) if use_weight_for_pathing_ else self.calculate_graph_distance(source_node, target_node, **graph_distance_finder_params)

                graph_distance = graph_distance_weight * graph_distance

                distance = graph_distance # тут возможно нужны какие-то множители
                if min_dist is None or distance < min_dist:
                    min_dist = distance

                    if min_dist == 0:
                        break # меньше 0 все равно не будет, чего считать лишнее

            if min_dist is None:
                min_dist = 1
            distances.append(min_dist)
        
        if len(distances) == 0:
            return zero_overlay_result

        return np.array(distances).mean()
        
    def estimate_gap_combined_distance(self, lhs_text, rhs_text, 
                                       trace = False, 
                                       subgraph_inflation = None, # до какой глубины соседние вершины будут добавляться в подграф
                                       use_weight_for_pathing = None, # гораздо медленнее, но в теории лучше

                                       graph_distance_weight = 1, # значение от 0 до 2, на него домножается расстояние по графу
                                       link_distance_weight = 1, # значение от 0 до 2, на него домножается ссылочное расстояние

                                       graph_distance_finder_params = {},
                                       link_distance_finder_params = {}, # веса, которые передаются функции поиска ссылочного расстояния для точной настройки

                                       zero_overlay_result = 0,

                                       node_detection_method = 'native' # native или parser, native работает быстро, но не поддерживает многословные вершины
                                       ): 
        
        use_weight_for_pathing_ = use_weight_for_pathing if use_weight_for_pathing else self.use_weight_for_pathing
        subgraph_inflation_ = subgraph_inflation if subgraph_inflation else self.subgraph_inflation
        trace_ = trace if trace else self.trace
        
        if node_detection_method == 'native':
        # старый способ извлечения вершин из текста
            lhs_words = lhs_text.split()
            rhs_words = rhs_text.split()

            lhs_core_graph_nodes = self.clear_node_list(lhs_words)
            rhs_core_graph_nodes = self.clear_node_list(rhs_words)
        else:
            lhs_core_graph_nodes = []
            for m in self.nodes_parser.findall(lhs_text):
                tokens_norm = [x.value for x in m.tokens]
                lhs_core_graph_nodes.append(" ".join(tokens_norm))
                
            rhs_core_graph_nodes = []
            for m in self.nodes_parser.findall(rhs_text):
                tokens_norm = [x.value for x in m.tokens]
                rhs_core_graph_nodes.append(" ".join(tokens_norm))

        lhs_graph = self.select_subgraph(lhs_core_graph_nodes, radius = subgraph_inflation_)
        rhs_graph = self.select_subgraph(rhs_core_graph_nodes, radius = subgraph_inflation_)

        lhs_graph_nodes = lhs_graph.nodes
        rhs_graph_nodes = rhs_graph.nodes

        if trace_:
            print(f"lens: \t {len(lhs_core_graph_nodes)} {len(rhs_core_graph_nodes)}")
            print(f"lens: \t {len(lhs_graph.nodes.keys())} {len(rhs_graph.nodes.keys())}")
            print("=========================================")
            print(f" \t {lhs_core_graph_nodes} {rhs_core_graph_nodes}")
            print(f" \t {lhs_graph.nodes.keys()} {rhs_graph.nodes.keys()}")

        # меряем расстояния от каждогй вершины графа 1 до ближайшей вершины графа 2
        distances = []
        for source_node in lhs_graph_nodes:
            min_dist = None
            for target_node in rhs_graph_nodes:
                if graph_distance_weight != 0:
                    graph_distance = self.calculate_graph_weight(source_node, target_node, **graph_distance_finder_params) if use_weight_for_pathing_ else self.calculate_graph_distance(source_node, target_node, **graph_distance_finder_params)
                else:
                    graph_distance = 0
                if link_distance_weight != 0:
                    link_distance = self.calculate_link_distance(source_node, target_node, **link_distance_finder_params)
                else:
                    link_distance = 0

                graph_distance = graph_distance_weight * graph_distance
                link_distance = link_distance_weight * link_distance

                distance = graph_distance + link_distance # тут возможно нужны какие-то множители
                if min_dist is None or distance < min_dist:
                    min_dist = distance

                    if min_dist == 0:
                        break # меньше 0 все равно не будет, чего считать лишнее

            if min_dist is None: # ! если все веса = 0, то используется только это - отображается ли одно из предложений в пустой подграф или нет
                                 # этого достаточно для получения 0.58 windiff
                min_dist = zero_overlay_result
            distances.append(min_dist)
        
        if len(distances) == 0:
            return zero_overlay_result

        return np.array(distances).mean() # от 0 до max(zero_overlay_result, (graph_distance_weight*max(0.5,no_path_weight) + link_distance_weight*0.5))
    

    
    def estimate_gap_lm_only(self, lhs_text, rhs_text): # определение схожести выполняется при помощи косинусного расстояния между эмбедингами предложений
        diff = self.calculate_cosine_difference(lhs_text, rhs_text)
        return diff # от 0 до 1
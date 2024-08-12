import re
import numpy as np
from tqdm import tqdm
from razdel import sentenize


def convert_sentence(text, terms_parser, term_to_idx):
    text_clean = re.sub("\W+|\d+|_+", " ", text)
    count_embedding = np.zeros((len(term_to_idx), ))
    for m in terms_parser.findall(text_clean):
        tokens_norm = [x.normalized for x in m.tokens]
        term = " ".join(tokens_norm)
        if term in term_to_idx:
            pattern_idx = term_to_idx[term]
            count_embedding[pattern_idx] += 1
    count_embedding = count_embedding.reshape((1, -1))
    return count_embedding


def convert_text_to_subgraphs(text, terms_parsers, term_to_idx):
    if isinstance(text, str):
        sentences = [item.text for item in sentenize(text)]
    elif isinstance(text, list):
        sentences = text
    subgraphs = []
    for i, sentence in enumerate(sentences):
        try:
            subgraphs.append(convert_sentence(sentence, terms_parsers, term_to_idx))
        except Exception as e:
            print(i)
            print(sentence)
            raise e
    subgraphs = np.concatenate(subgraphs, axis=0)
    return subgraphs, sentences


def separate_complementary_graph(subgraph, complementary_graph, adjacency_table):
    neighbour_nodes = []
    distant_nodes = []
    indices = np.arange(len(complementary_graph))
    subgraph_ids = indices[subgraph > 0]
    comp_graph_ids = indices[complementary_graph > 0]
    for idx in comp_graph_ids:
        if idx in subgraph_ids:
            neighbour_nodes.append(idx)
        else:
            is_neighbour = False
            for sub_idx in subgraph_ids:
                if (adjacency_table[idx, sub_idx] > 0 
                    or adjacency_table[sub_idx, idx] > 0):
                    is_neighbour = True
                    break
            if is_neighbour:
                neighbour_nodes.append(idx)
            else:
                distant_nodes.append(idx)
    return {"neighbour_nodes": neighbour_nodes, "distant_nodes": distant_nodes}


def compute_cohesion_gain(complementary_subgraph, separation):
    neighbour_cohesion = complementary_subgraph[separation["neighbour_nodes"]].sum()
    distant_cohesion = complementary_subgraph[separation["distant_nodes"]].sum()
    return neighbour_cohesion - distant_cohesion


def estimate_gaps(subgraphs, adjacency_table):
    gains = []
    for i in range(len(subgraphs)-1):
        separation_1 = separate_complementary_graph(
            subgraph=subgraphs[i], 
            complementary_graph=subgraphs[i+1],
            adjacency_table=adjacency_table
        )
        cohesion_gain_1 = compute_cohesion_gain(
            complementary_subgraph=subgraphs[i+1],
            separation=separation_1
        )
        separation_2 = separate_complementary_graph(
            subgraph=subgraphs[i+1], 
            complementary_graph=subgraphs[i],
            adjacency_table=adjacency_table
        )
        cohesion_gain_2 = compute_cohesion_gain(
            complementary_subgraph=subgraphs[i],
            separation=separation_2
        )
        cohesion_gain = 0.5*(cohesion_gain_1+cohesion_gain_2)
        gains.append(cohesion_gain)
    gains = np.array(gains)
    return gains
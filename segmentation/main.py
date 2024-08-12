import re
import os
import pickle
import numpy as np
from tqdm import tqdm

import yargy
from yargy.pipelines import morph_pipeline
from razdel import sentenize

# from utils import convert_text_to_subgraphs, estimate_gaps
import sys
sys.path.append("..")
from segmentation.utils import convert_text_to_subgraphs, estimate_gaps


def segment_text(
    path_to_adjacency_table, 
    path_to_terms_table, 
    source_path, 
    destination_path
):
    with open(path_to_terms_table, "rb") as f:
        table = pickle.load(f)
    adjacency_table = np.load(path_to_adjacency_table)
    terms = [item["term"].strip() for item in table]
    tokenizer = yargy.tokenizer.MorphTokenizer()
    terms_norm = [" ".join([x.normalized for x in tokenizer(item)])
                for item in terms]
    term_to_idx = {term: i for i, term in enumerate(terms_norm)}
    terms_parser = yargy.Parser(morph_pipeline(terms_norm))

    with open(source_path, "r") as f:
        text = f.read()
    subgraphs = convert_text_to_subgraphs(text, terms_parser, term_to_idx)
    gains = estimate_gaps(subgraphs, adjacency_table)

    sentences = []
    for s in sentenize(text):
        sentence_prep = [x.strip() for x in s.text.split("\n") if len(x.strip()) > 0]
        sentence_prep = "\n".join(sentence_prep)
        sentences.append(sentence_prep)
    threshold = gains.mean()-gains.std()
    break_indices = np.arange(len(gains))[gains <= threshold]
    segments = []
    for i in range(len(break_indices)):
        if i == 0:
            r = 0
            l = break_indices[i]+1
        elif i == len(break_indices)-1:
            r = break_indices[i]+1
            l = len(sentences)
        else:
            r = break_indices[i]+1
            l = break_indices[i+1]+1
        segments.append(sentences[r : l])
    with open(destination_path, "w") as f:
        for i, x in enumerate(segments):
            f.write(" ".join(x))
            if i != len(segments)-1:
                f.write("\n")
                f.write("-"*10)
                f.write("\n")


def segment_texts(path_to_adjacency_table, path_to_terms_table, root_to_texts, root_to_save):
    with open(path_to_terms_table, "rb") as f:
        table = pickle.load(f)
    adjacency_table = np.load(path_to_adjacency_table)
    terms = [item["term"].strip() for item in table]
    tokenizer = yargy.tokenizer.MorphTokenizer()
    terms_norm = [" ".join([x.normalized for x in tokenizer(item)])
                for item in terms]

    term_to_idx = {term: i for i, term in enumerate(terms_norm)}
    terms_parser = yargy.Parser(morph_pipeline(terms_norm))
    for i, item in enumerate(tqdm(sorted(os.listdir(root_to_texts)))):
        if i == 37:
            print(item)
        path_to_text = os.path.join(root_to_texts, item)
        with open(path_to_text, "r") as f:
            text = f.read()
        subgraphs = convert_text_to_subgraphs(text, terms_parser, term_to_idx)
        gains = estimate_gaps(subgraphs, adjacency_table)

        sentences = []
        for s in sentenize(text):
            sentence_prep = [x.strip() for x in s.text.split("\n") if len(x.strip()) > 0]
            sentence_prep = "\n".join(sentence_prep)
            sentences.append(sentence_prep)
        threshold = gains.mean()-gains.std()
        break_indices = np.arange(len(gains))[gains <= threshold]
        segments = []
        for i in range(len(break_indices)):
            if i == 0:
                r = 0
                l = break_indices[i]+1
            elif i == len(break_indices)-1:
                r = break_indices[i]+1
                l = len(sentences)
            else:
                r = break_indices[i]+1
                l = break_indices[i+1]+1
            segments.append(sentences[r : l])
        destination_path = os.path.join(root_to_save, item)
        with open(destination_path, "w") as f:
            for i, x in enumerate(segments):
                f.write(" ".join(x))
                if i != len(segments)-1:
                    f.write("\n")
                    f.write("-"*10)
                    f.write("\n")
        # break
                    

class PartitionSolver:
    def __init__(self,
                 path_to_adjacency_table: str,
                 path_to_table: str,
                 normalized_terms: str = False):
        self.adjacency_table = np.load(path_to_adjacency_table)
        with open(path_to_table, "rb") as f:
            table = pickle.load(f)
        terms = [item["term"].strip() for item in table]
        tokenizer = yargy.tokenizer.MorphTokenizer()
        if normalized_terms:
            terms_norm = terms
        else:
            terms_norm = [" ".join([x.normalized for x in tokenizer(item)])
                        for item in terms]

        self.term_to_idx = {term: i for i, term in enumerate(terms_norm)}
        self.terms_parser = yargy.Parser(morph_pipeline(terms_norm))

    def solve_partition(self, text: str, scoring: str = "base"):
        subgraphs, _ = convert_text_to_subgraphs(text, self.terms_parser, self.term_to_idx)
        gains = estimate_gaps(subgraphs, self.adjacency_table)
        if scoring == "base":
            depth_scores = gains
            threshold = depth_scores.mean()-depth_scores.std()
        elif scoring == "texttiling":
            depth_scores = list()
            for idx, _ in enumerate(gains):
                ds = compute_ds(gains, idx)
                depth_scores.append(ds)
            depth_scores = np.array(depth_scores)
            threshold = depth_scores.mean()-depth_scores.std()/2
        prediction = ""
        for i in range(len(depth_scores)):
            if scoring == "base":
                if depth_scores[i] < threshold:
                    prediction += "1"
                else:
                    prediction += "0"
            elif scoring == "texttiling":
                if depth_scores[i] > threshold:
                    prediction += "1"
                else:
                    prediction += "0"
        prediction += "1"
        depth_scores = list(depth_scores)
        depth_scores.append(-1)
        return prediction, depth_scores, threshold
    

def compute_ds(a: np.ndarray, idx: int):
    if idx == 0:
        H_r = a[idx]
        r = idx
        while (r < len(a)-1 and a[r+1] >= H_r):
            H_r = a[r+1]
            r += 1
        depth_score = H_r - a[idx]
    elif idx == len(a)-1:
        H_l = a[idx]
        l = idx
        while (l > -1 and a[l-1] <= H_l):
            H_l = a[l-1]
            l -= 1
        depth_score = H_l - a[idx]
    else:
        H_r, H_l = a[idx], a[idx]
        l, r = idx, idx
        while (r < len(a)-1 and a[r+1] >= H_r):
            H_r = a[r+1]
            r += 1
        while (l > -1 and a[l-1] <= H_l):
            H_l = a[l-1]
            l -= 1
        depth_score = (H_r + H_l)/2 - a[idx]
    return depth_score


if __name__ == "__main__":
    with open("./data/table.pkl", "rb") as f:
        table = pickle.load(f)
    adjacency_table = np.load("./data/adjacency_table.npy")
    terms = [item["term"].strip() for item in table]
    tokenizer = yargy.tokenizer.MorphTokenizer()
    terms_norm = [" ".join([x.normalized for x in tokenizer(item)])
                for item in terms]

    term_to_idx = {term: i for i, term in enumerate(terms_norm)}
    terms_parser = yargy.Parser(morph_pipeline(terms_norm))
    
    path_to_text = "../data/prnd_ru_txt/Рудько_0.txt"
    with open(path_to_text, "r") as f:
        text = f.read()

    subgraphs, sentences = convert_text_to_subgraphs(text, terms_parser, term_to_idx)
    gains = estimate_gaps(subgraphs, adjacency_table)

    sentences = []
    for item in sentenize(text):
        sentence_prep = [x.strip() for x in item.text.split("\n") if len(x.strip()) > 0]
        sentence_prep = "\n".join(sentence_prep)
        sentences.append(sentence_prep)
    
    with open(os.path.join(".", f"{os.path.basename(path_to_text)[:-4]}.txt"), "w") as f:
        threshold = gains.mean()-gains.std()
        for i in range(len(gains)):
            f.write(sentences[i])
            f.write("\n")
            f.write("-"*10)
            f.write(f"  {str(gains[i])}  ")
            f.write("-"*10)
            if gains[i] < threshold:
                f.write(f"  SEGMENT BREAK  ")
            f.write("\n")
        f.write(sentences[-1])
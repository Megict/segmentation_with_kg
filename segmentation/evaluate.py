import os
import json
import numpy as np
from tqdm import tqdm
from nltk.metrics import pk, windowdiff


from main import PartitionSolver


if __name__ == "__main__":
    path_to_adjacency_table = "/home/artur/work/aspir/build_graph/adjacency_table.npy"
    path_to_table = "/home/artur/work/aspir/build_graph/table.pkl"
    TABLE = "Словарь_М"
    # path_to_adjacency_table = "/home/artur/work/aspir/build_graph/adjacency_table_2.npy"
    # path_to_table = "/home/artur/work/aspir/build_graph/table_2.pkl"
    # TABLE = "Словарь_Б"
    # path_to_adjacency_table = "/home/artur/work/aspir/segmentation_forces/distance_table_3.npy"
    # path_to_table = "/home/artur/work/aspir/segmentation_forces/table_3.pkl"
    # TABLE = "Граф_Макс"

    path_to_data = "/home/artur/work/aspir/data/segmentation_dataset"
    SCORING = "base" #base, texttiling

    partition_solver = PartitionSolver(
        path_to_adjacency_table, path_to_table, normalized_terms=False
    )
    Pks = list()
    WinDiffs = list()

    for item in tqdm(os.listdir(path_to_data)):
        filepath = os.path.join(path_to_data, item)
        text, y_true = list(), ""
        with open(filepath, "r") as f:
            lines = f.read().split("\n")
            for i, x in enumerate(lines):
                if len(x) == 0 or x == "--- SEGMENT BRAKE ---":
                    continue
                if i < len(lines)-1:
                    if lines[i+1] == "--- SEGMENT BRAKE ---":
                        y_true += "1"
                    else:
                        y_true += "0"
                text.append(x+".")
        y_pred, depth_scores, threshold = partition_solver.solve_partition(text, scoring=SCORING)

        print(y_true)
        print(y_pred)

        Pk = pk(y_true, y_pred, k=3)
        WinDiff = windowdiff(y_true, y_pred, k=3)

        filename = os.path.basename(filepath)
        with open(f"./results/{filename}", "w") as f:
            f.write(f"Pk: {Pk}, WinDiff: {WinDiff}\n\n")
            f.write(f"threshold: {threshold}")
            for line, label_pred, ds in zip(text, y_pred, depth_scores):
                f.write(f"{str(ds)} --- " + line + "\n")
                if label_pred == "1":
                    f.write(" --- SEGMENT BREAK --- \n")

        Pks.append(Pk)
        WinDiffs.append(WinDiff)
    Pk, WinDiff = np.mean(Pks), np.mean(WinDiffs)
    with open("./exp_logs.jsonl", "r") as f:
        n_lines = len(f.readlines())
    with open("./exp_logs.jsonl", "a") as f:
        result = {
            "idx": n_lines,
            "Pk": Pk,
            "WinDiff": WinDiff,
            "graph": TABLE,
            "scoring": SCORING
        }
        f.write("\n")
        f.write(json.dumps(result, ensure_ascii=False))
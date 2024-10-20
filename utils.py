from collections import defaultdict
import numpy as np
import torch
from torch_geometric.data import Data


def custom_tokenizer(text):
    return text


def generateGraphFromSequence(sequence, embedding_model, label):
    node_map = {}
    node_features = []
    edges_with_frequencies = defaultdict(int)

    for i in range(len(sequence)):
        current_string = sequence[i]

        if current_string not in node_map:
            node_index = len(node_map)
            node_map[current_string] = node_index
            node_features.append(embedding_model.getEmbeddedString(current_string))

        if i > 0:
            previous_string = sequence[i - 1]
            src = node_map[previous_string]
            dst = node_map[current_string]
            edges_with_frequencies[(src, dst)] += 1

    edge_index = torch.tensor(list(edges_with_frequencies.keys()), dtype=torch.int32).t().contiguous()
    edge_attr = torch.tensor(list(edges_with_frequencies.values()), dtype=torch.float).unsqueeze(1)
    node_features = torch.tensor(node_features, dtype=torch.float)

    #check if there's no edges; in this case a self loop will be added to the only node existing
    if edge_index.numel() == 0:
        edge_index = torch.tensor([[0], [0]], dtype=torch.int32)
        edge_attr = torch.tensor([[0.0]], dtype=torch.float)

    return Data(x=node_features, y=torch.tensor([label], dtype=torch.float), edge_index=edge_index, edge_attr=edge_attr)


def generateGraphsFromJson(json_example, embedding_model, label):
    graphs = []
    for macro_category in json_example.keys():
        for level_one_action in json_example[macro_category].keys():
            graphs.append(generateGraphFromSequence(json_example[macro_category][level_one_action],
                                                    embedding_model, label))
    return graphs


def convertTimeDelta(before, after):
    tot = after - before
    d = np.timedelta64(tot, 'D')
    tot -= d
    h = np.timedelta64(tot, 'h')
    tot -= h
    m = np.timedelta64(tot, 'm')
    tot -= m
    s = np.timedelta64(tot, 's')
    tot -= s
    ms = np.timedelta64(tot, 'ms')
    out = str(d) + " " + str(h) + " " + str(m) + " " + str(s) + " " + str(ms)
    return out

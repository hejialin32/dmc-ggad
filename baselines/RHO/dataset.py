import numpy as np
import torch
import scipy.io as sio
import scipy.sparse as sp
import networkx as nx
from numpy import inf
from scipy.sparse import csgraph
import os
import importlib.util

_base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_utils_path = os.path.join(_base_dir, 'utils', 'utils.py')
_spec = importlib.util.spec_from_file_location("shared_utils", _utils_path)
_shared_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_shared_utils)
load_mat = _shared_utils.load_mat


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)


def _load_mat_data(name):
    """从 .mat 文件加载数据集，返回 (features, labels, adj)"""
    data = load_mat(name, train_rate=0.3, val_rate=0.1)
    adj = data[0]
    feat = data[1]
    ano_labels = data[2]
    features = torch.tensor(feat.toarray(), dtype=torch.float32)
    labels = torch.tensor(ano_labels, dtype=torch.long).squeeze(-1)
    return features, labels, adj


def _compute_lap_from_adj(adj, num_nodes):
    """从邻接矩阵计算归一化拉普拉斯矩阵"""
    adj = adj.maximum(adj.T)
    Lap = csgraph.laplacian(adj, normed=True)
    return Lap


def _load_lap_matrix(name, adj, num_nodes):
    """尝试加载预计算的拉普拉斯矩阵，失败则从邻接矩阵计算"""
    lap_paths = [
        os.path.abspath('./datasets/Lap_matrix_{}.npz'.format(name)),
        os.path.abspath('../../datasets/Lap_matrix_{}.npz'.format(name)),
        os.path.abspath('../../datasets/laplacian_matrices/Lap_matrix_{}.npz'.format(name)),
        'C:\\Users\\27586\\Desktop\\GraphAD-Framework\\datasets\\laplacian_matrices\\Lap_matrix_{}.npz'.format(name),
        r'C:\Users\27586\Desktop\GraphAD-Framework\BaseLine_New\3_RHO\datasets\Lap_matrix_{}.npz'.format(name),
    ]
    for path in lap_paths:
        try:
            Lap = sp.load_npz(path)
            return Lap
        except Exception:
            continue
    Lap = _compute_lap_from_adj(adj, num_nodes)
    return Lap


class Dataset:
    def __init__(self, name='tfinance', homo=True, anomaly_alpha=None, anomaly_std=None):
        self.name = name
        features = None
        labels = None
        adj = None

        if name == 'tfinance':
            label, feat, adj = load_mat('t_finance', train_rate=0.3, val_rate=0.1)[:3]
            features = torch.tensor(feat.toarray(), dtype=torch.float32)
            labels = torch.tensor(label.squeeze(), dtype=torch.long)
            if anomaly_std is not None:
                feat_np = features.numpy()
                anomaly_id = labels.nonzero().squeeze(1).numpy()
                feat_np = (feat_np - np.average(feat_np, 0)) / np.std(feat_np, 0)
                feat_np[anomaly_id] = anomaly_std * feat_np[anomaly_id]
                features = torch.tensor(feat_np)
            if anomaly_alpha is not None:
                feat_np = features.numpy()
                anomaly_id = list(labels.nonzero().squeeze(1).numpy())
                normal_id = list((labels == 0).nonzero().squeeze(1).numpy())
                diff = anomaly_alpha * len(labels) - len(anomaly_id)
                import random
                new_id = random.sample(normal_id, int(diff))
                for idx in new_id:
                    aid = random.choice(anomaly_id)
                    feat_np[idx] = feat_np[aid]
                    labels[idx] = 1
                features = torch.tensor(feat_np)

        elif name == 'amazon':
            features, labels, adj = _load_mat_data(name)

        elif name in ['reddit', 'photo', 'elliptic']:
            features, labels, adj = _load_mat_data(name)

        elif name in ['dblp','cora', 'citeseer', 'pubmed', 'facebook', 'blogcatalog', 'flickr', 'acm', 'yelpchi']:
            features, labels, adj = _load_mat_data(name)

        elif name in ['tolokers', 'questions']:
            from torch_geometric.datasets import HeterophilousGraphDataset
            dataset = HeterophilousGraphDataset(root="./datasets/", name=name)
            graph_data = dataset[0]
            features = graph_data.x.float()
            labels = graph_data.y.data.long().squeeze(-1)
            edge_index = graph_data.edge_index
            num_nodes = graph_data.num_nodes
            adj = sp.coo_matrix(
                (np.ones(edge_index.shape[1]), (edge_index[0].numpy(), edge_index[1].numpy())),
                shape=(num_nodes, num_nodes)
            ).tocsr()

        elif name == 'dgraphfin':
            features, labels, adj = _load_mat_data(name)

        else:
            print('no such dataset')
            raise Exception('no such dataset')

        if name not in ['tolokers', 'questions']:
            labels = labels.long().squeeze(-1)
            features = features.float()

        Lap = _load_lap_matrix(name, adj, features.shape[0])
        Lap = sparse_mx_to_torch_sparse_tensor(Lap)

        self.features = features
        self.labels = labels
        self.Lap = Lap
import os
import pickle
import random

import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import torch


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    return torch.sparse_coo_tensor(indices, values, torch.Size(sparse_mx.shape))


def preprocess_features(features):
    rowsum = np.asarray(features.sum(1)).reshape(-1)
    rowsum[rowsum == 0] = 1.0
    r_inv = np.power(rowsum, -1)
    return sp.diags(r_inv).dot(features)


def normalize_adj(adj):
    rowsum = np.asarray(adj.sum(1)).reshape(-1)
    rowsum[rowsum == 0] = 1.0
    d_inv_sqrt = np.power(rowsum, -0.5)
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def _find_dataset_path(dataset, file_path=None):
    if file_path is not None:
        if os.path.isdir(file_path):
            for ext in ('.mat', '.pkl', '.npz'):
                path = os.path.join(file_path, f'{dataset}{ext}')
                if os.path.exists(path):
                    return path
        return file_path

    if os.path.exists(dataset):
        return dataset

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = []
    for base in (
        os.path.join(root, 'datasets'),
        os.path.join(os.path.dirname(root), 'datasets'),
        os.path.join(os.path.dirname(os.path.dirname(root)), 'datasets'),
        os.path.join(os.getcwd(), 'datasets'),
    ):
        for ext in ('.mat', '.pkl', '.npz'):
            candidates.append(os.path.join(base, f'{dataset}{ext}'))

    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f'Dataset not found: {dataset}')


def _first_present(data, keys, default=None):
    for key in keys:
        if key in data:
            return data[key]
    return default


def _load_raw_dataset(path):
    if path.endswith('.mat'):
        data = sio.loadmat(path)
    elif path.endswith('.npz'):
        npz_data = np.load(path, allow_pickle=True)
        data = dict(npz_data)

        adj_keys = ('adj', 'adjacency', 'A', 'network', 'Network')
        feat_keys = ('node_features', 'features', 'X', 'attr', 'attribute', 'Attributes', 'feat', 'feature')
        label_keys = ('node_labels', 'label', 'labels', 'Label', 'gnd', 'y')

        adj = _first_present(data, adj_keys)
        feat = _first_present(data, feat_keys)
        labels = _first_present(data, label_keys)

        if adj is None:
            edge_index = _first_present(data, ('edges', 'edge_index'))
            if edge_index is not None:
                edge_index = np.asarray(edge_index)
                if feat is not None:
                    n = np.asarray(feat).shape[0]
                elif labels is not None:
                    n = np.asarray(labels).shape[0]
                else:
                    n = int(np.max(edge_index)) + 1
                adj = np.zeros((n, n), dtype=np.float32)
                if edge_index.shape[0] == 2:
                    valid = (edge_index[0] < n) & (edge_index[1] < n)
                    adj[edge_index[0, valid], edge_index[1, valid]] = 1.0
                    adj[edge_index[1, valid], edge_index[0, valid]] = 1.0
                else:
                    valid = (edge_index[:, 0] < n) & (edge_index[:, 1] < n)
                    idx = edge_index[valid]
                    adj[idx[:, 0], idx[:, 1]] = 1.0
                    adj[idx[:, 1], idx[:, 0]] = 1.0

        if adj is None or feat is None or labels is None:
            raise ValueError(f'Dataset is missing adjacency, features, or labels: {path}')
        return adj, feat, labels
    elif path.endswith('.pkl'):
        with open(path, 'rb') as f:
            try:
                data = pickle.load(f)
            except Exception:
                f.seek(0)
                import joblib
                data = joblib.load(f)
    else:
        raise ValueError(f'Unsupported dataset format: {path}')

    if isinstance(data, tuple) and len(data) >= 3:
        adj, feat, labels = data[:3]
        return adj, feat, labels

    if 'PAP' in data and 'PLP' in data and 'PTP' in data:
        adj = data['PAP']
    elif 'homo' in data:
        adj = data['homo']
    elif 'adj_matrix' in data:
        adj = data['adj_matrix']
    elif 'net_APA' in data:
        adj = data['net_APA']
    elif 'MAM' in data:
        adj = data['MAM']
    else:
        adj = _first_present(data, ('Network', 'A', 'adj', 'adjacency', 'network'))

    feat = _first_present(data, ('Attributes', 'X', 'features', 'feature', 'node_features', 'feat'))
    labels = _first_present(data, ('Label', 'gnd', 'label', 'labels', 'node_labels', 'y'))

    if adj is None or feat is None or labels is None:
        raise ValueError(f'Dataset is missing adjacency, features, or labels: {path}')
    return adj, feat, labels


def _as_1d_labels(labels):
    labels = np.squeeze(np.asarray(labels))
    if labels.ndim > 1:
        labels = labels[0] if labels.shape[0] == 1 else labels[:, 0]
    return labels.astype(int)


def load_mat(dataset, train_rate=0.25, file_path=None):
    path = _find_dataset_path(dataset, file_path)
    adj, feat, labels = _load_raw_dataset(path)

    adj = sp.csr_matrix(adj)
    feat = sp.lil_matrix(feat)
    labels = _as_1d_labels(labels)

    all_idx = list(range(adj.shape[0]))
    normal_idx = [idx for idx in all_idx if labels[idx] == 0]
    abnormal_idx = [idx for idx in all_idx if labels[idx] == 1]
    random.shuffle(normal_idx)
    random.shuffle(abnormal_idx)

    num_train = int(len(normal_idx) * train_rate)
    if train_rate > 0 and num_train < 1 and normal_idx:
        num_train = 1

    idx_train = normal_idx[:num_train]
    idx_val = []
    idx_test = normal_idx[num_train:] + abnormal_idx
    random.shuffle(idx_train)
    random.shuffle(idx_test)

    normal_label_idx = idx_train[: int(len(idx_train) * 0.5)]
    random.shuffle(normal_label_idx)
    abnormal_label_idx = normal_label_idx[: int(len(normal_label_idx) * 0.15)]

    return (
        adj,
        feat,
        labels,
        all_idx,
        idx_train,
        idx_val,
        idx_test,
        labels,
        None,
        None,
        normal_label_idx,
        abnormal_label_idx,
    )

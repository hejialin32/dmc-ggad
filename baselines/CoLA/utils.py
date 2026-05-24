import numpy as np
import scipy.sparse as sp
import torch
import scipy.io as sio
import random
import os


def _load_data_file(dataset):
    dataset_root = os.environ.get("DATASET_DIR", "/root/dataset")
    candidates = [
        os.path.join(dataset_root, f"{dataset}.mat"),
        os.path.join(dataset_root, f"{dataset.lower()}.mat"),
        os.path.join(dataset_root, f"{dataset}.npz"),
        os.path.join(dataset_root, f"{dataset.lower()}.npz"),
        os.path.join("./dataset", f"{dataset}.mat"),
        os.path.join("./dataset", f"{dataset}.npz"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


def sparse_to_tuple(sparse_mx, insert_batch=False):
    """Convert sparse matrix to tuple representation."""
    """Set insert_batch=True if you want to insert a batch dimension."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        if insert_batch:
            coords = np.vstack((np.zeros(mx.row.shape[0]), mx.row, mx.col)).transpose()
            values = mx.data
            shape = (1,) + mx.shape
        else:
            coords = np.vstack((mx.row, mx.col)).transpose()
            values = mx.data
            shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense(), sparse_to_tuple(features)

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset+labels_dense.ravel()] = 1
    return labels_one_hot

def load_mat(dataset, train_rate=0.3, val_rate=0.0):
    """Load .mat dataset."""
    data_path = _load_data_file(dataset)
    if data_path.endswith(".npz"):
        npz_data = np.load(data_path)
        data = {key: npz_data[key] for key in npz_data.files}
    else:
        data = sio.loadmat(data_path)

    label = data['Label'] if ('Label' in data) else data['gnd'] if ('gnd' in data) else data['node_labels']
    attr = data['Attributes'] if ('Attributes' in data) else data['X'] if ('X' in data) else data['node_features']
    if 'Network' in data:
        network = data['Network']
    elif 'A' in data:
        network = data['A']
    elif 'edges' in data:
        edges = np.asarray(data['edges'])
        if edges.ndim != 2:
            raise ValueError(f"Unsupported edges format: {edges.shape}")
        if edges.shape[0] == 2 and edges.shape[1] != 2:
            src, dst = edges[0], edges[1]
        else:
            src, dst = edges[:, 0], edges[:, 1]
        num_nodes = np.asarray(attr).shape[0]
        rows = np.concatenate([src, dst]).astype(np.int64)
        cols = np.concatenate([dst, src]).astype(np.int64)
        values = np.ones(len(rows), dtype=np.float32)
        network = sp.coo_matrix((values, (rows, cols)), shape=(num_nodes, num_nodes))
    else:
        raise KeyError("Network/A/edges key not found in dataset")

    adj = sp.csr_matrix(network)
    feat = sp.lil_matrix(attr)

    ano_labels = np.squeeze(np.array(label))
    if 'Class' in data:
        labels = np.squeeze(np.array(data['Class'], dtype=np.int64) - 1)
        num_classes = np.max(labels) + 1
        labels = dense_to_one_hot(labels, num_classes)
    else:
        labels = dense_to_one_hot(ano_labels.astype(np.int64), 2)
    if 'str_anomaly_label' in data:
        str_ano_labels = np.squeeze(np.array(data['str_anomaly_label']))
        attr_ano_labels = np.squeeze(np.array(data['attr_anomaly_label']))
    else:
        str_ano_labels = None
        attr_ano_labels = None

    num_node = adj.shape[0]
    num_train = int(num_node * train_rate)
    num_val = int(num_node * val_rate)
    all_idx = list(range(num_node))
    random.shuffle(all_idx)
    idx_train = all_idx[ : num_train]
    idx_val = all_idx[num_train : num_train + num_val]
    idx_test = all_idx[num_train + num_val : ]

    return adj, feat, labels, idx_train, idx_val, idx_test, ano_labels, str_ano_labels, attr_ano_labels

def generate_rwr_subgraph(adj, subgraph_size, neighbors=None):
    """Generate subgraph with RWR algorithm using numpy (no DGL dependency)."""
    adj_mat = sp.csr_matrix(adj)
    n_nodes = adj_mat.shape[0]
    reduced_size = subgraph_size - 1

    # 预计算每个节点的邻居列表，允许外部传入缓存结果
    if neighbors is None:
        neighbors = [adj_mat[i].indices for i in range(n_nodes)]

    subv = []
    for i in range(n_nodes):
        # 第一次尝试：restart_prob=0.99, length=subgraph_size*3
        trace = [i]
        current = i
        for _ in range(subgraph_size * 3):
            if np.random.random() < 0.99:
                current = i
            else:
                if len(neighbors[current]) > 0:
                    current = np.random.choice(neighbors[current])
            trace.append(current)

        subv_i = list(set(trace))

        # 如果节点不够，重试
        retry_time = 0
        while len(subv_i) < reduced_size:
            trace = [i]
            current = i
            for _ in range(subgraph_size * 5):
                if np.random.random() < 0.9:
                    current = i
                else:
                    if len(neighbors[current]) > 0:
                        current = np.random.choice(neighbors[current])
                trace.append(current)

            subv_i = list(set(trace))
            retry_time += 1
            if len(subv_i) <= 2 and retry_time > 10:
                subv_i = (subv_i * reduced_size)

        subv_i = subv_i[:reduced_size]
        subv_i.append(i)
        subv.append(subv_i)

    return subv


def adj_to_dgl_graph(adj):
    """Convert adjacency matrix to a DGL graph."""
    import dgl
    import networkx as nx
    try:
        nx_graph = nx.from_scipy_sparse_array(adj)
    except AttributeError:
        nx_graph = nx.from_scipy_sparse_matrix(adj)
    return dgl.from_networkx(nx_graph)

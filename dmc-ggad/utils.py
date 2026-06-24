import os
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import scipy.sparse as sp
import torch
import scipy.io as sio
import random

# import dgl
from collections import Counter
import time
import os


import scipy.io as sio


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Sparse mx to torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)


def is_sparse_matrix(matrix):
    """Is sparse matrix."""
    return hasattr(matrix, 'tocoo') or (isinstance(matrix, torch.Tensor) and matrix.is_sparse)


def save_fake_anomaly_dataset(adj, features, normal_idx, fake_anomaly_idx, dataset_name, save_path):
    """Save fake anomaly dataset."""

    all_nodes = list(set(normal_idx + fake_anomaly_idx))
    all_nodes.sort()
    n = len(all_nodes)


    adj_sub = adj[all_nodes, :][:, all_nodes]

    features_sub = features[all_nodes]


    labels_sub = np.zeros(n, dtype=int)

    idx_map = {orig_idx: new_idx for new_idx, orig_idx in enumerate(all_nodes)}
    for orig_idx in fake_anomaly_idx:
        new_idx = idx_map[orig_idx]
        labels_sub[new_idx] = 1


    os.makedirs(save_path, exist_ok=True)
    save_file = f"{save_path}/{dataset_name}_fake_anomalies.mat"


    sio.savemat(
        save_file,
        {
            'Network': adj_sub,
            'Attributes': features_sub,
            'Label': labels_sub.reshape(1, -1),
            'original_indices': np.array(all_nodes),

            'fake_anomaly_original_indices': np.array(fake_anomaly_idx)
        }
    )
    print(f"\n[DatasetSaved] Saved: {save_file}")
    print(
        f"[Saved] total nodes: {n} (normal: {len(normal_idx)}, synthetic anomalies: {len(fake_anomaly_idx)})")
    print(f"[Saved] real anomalies are excluded\n")


def sparse_to_tuple(sparse_mx, insert_batch=False):
    """Sparse to tuple."""
    """ batch ,0"""

    def to_tuple(mx):

        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        if insert_batch:

            coords = np.vstack(
                (np.zeros(mx.row.shape[0]), mx.row, mx.col)).transpose()
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
    """Preprocess features."""
    rowsum = np.array(features.sum(1))

    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.

    r_mat_inv = sp.diags(r_inv)

    features = r_mat_inv.dot(features)
    return features.todense(), sparse_to_tuple(features)


def normalize_adj(adj):
    """Normalize adj."""
    adj = sp.coo_matrix(adj)

    rowsum = np.array(adj.sum(1))

    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.

    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def dense_to_one_hot(labels_dense, num_classes):
    """Dense to one hot."""
    num_labels = labels_dense.shape[0]

    index_offset = np.arange(num_labels) * num_classes

    labels_one_hot = np.zeros((num_labels, num_classes))

    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def load_mat(dataset, train_rate=0.25, val_rate=0.1, file_path=None):
    """Load mat."""
    import pickle


    if file_path is None:

        if dataset.endswith('.pkl'):
            actual_path = dataset
        elif dataset.endswith('.mat'):
            actual_path = dataset
        else:


            utils_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(utils_dir)
            parent_project_root = os.path.dirname(project_root)
            datasets_dir = os.path.join(parent_project_root, 'datasets')
            print(f"Dataset: {datasets_dir}")


            # user_datasets_dir = "C:\\Users\\27586\\Desktop\\Sync_Island\\datasets"

            possible_paths = [
                os.path.join(datasets_dir, f"{dataset}.mat"),
                os.path.join(datasets_dir, f"{dataset}.pkl"),
                os.path.join(datasets_dir, f"{dataset}.npz"),
                 f"../datasets/{dataset}.mat",
                f"../datasets/{dataset}.pkl",
                f"../datasets/{dataset}.npz",
                f"../../datasets/{dataset}.mat",
                f"../../datasets/{dataset}.pkl",
                f"../../datasets/{dataset}.npz",
                f"../../../datasets/{dataset}.mat",
                f"../../../datasets/{dataset}.pkl",
                f"../../../datasets/{dataset}.npz",
                f"/root/datasets/{dataset}.mat",
                f"/root/datasets/{dataset}.pkl",
                f"/root/datasets/{dataset}.npz",
                f"/root/dmc_-island/datasets/{dataset}.mat",
                f"/root/dmc_-island/datasets/{dataset}.pkl",
                f"/root/dmc_-island/datasets/{dataset}.npz",
                f"/root/dataset/{dataset}.mat",
                f"/root/dataset/{dataset}.pkl",
                f"/root/dataset/{dataset}.npz",
            ]
            actual_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    actual_path = path
                    break
            if actual_path is None:

                actual_path = possible_paths[0]
    else:

        if os.path.isdir(file_path):
            actual_path = os.path.join(file_path, f"{dataset}.mat")

            if not os.path.exists(actual_path):
                actual_path = os.path.join(file_path, f"{dataset}.npz")
        else:
            actual_path = file_path


    if actual_path.endswith('.pkl'):


        try:
            import joblib
            with open(actual_path, 'rb') as f:
                data = joblib.load(f)
        except (ImportError, Exception):

            with open(actual_path, 'rb') as f:
                data = pickle.load(f)


        if isinstance(data, tuple) and len(data) >= 3:

            adj, feat, label = data
            network = adj
            attr = feat
        elif isinstance(data, dict):

            label = data['Label'] if ('Label' in data) else data['gnd'] if ('gnd' in data) else data.get('label', None)

            attr = data['Attributes'] if ('Attributes' in data) else data['X'] if ('X' in data) else data.get('feat', None) if ('feat' in data) else data.get('features', None) if ('features' in data) else data.get('feature', None)

            network = data['Network'] if ('Network' in data) else data['A'] if ('A' in data) else data.get('adj', None) if ('adj' in data) else data.get('homo', None)
        else:
            raise ValueError(f" pkl : {type(data)}")
    elif actual_path.endswith('.mat'):

        data = sio.loadmat(actual_path)



        if 'PAP' in data and 'PLP' in data and 'PTP' in data:

            network = data['PAP']
        elif 'homo' in data:

            network = data['homo']
        elif 'adj_matrix' in data:

            network = data['adj_matrix']
        elif 'net_APTPA' in data or 'net_APCPA' in data or 'net_APA' in data:

            network = data['net_APA']
        elif 'MAM' in data or 'MDM' in data or 'MYM' in data:

            network = data['MAM']
        elif 'Network' in data:
            network = data['Network']
        elif 'A' in data:
            network = data['A']
        elif 'adj' in data:
            network = data['adj']
        else:
            print("Warning: ,")

            if 'Attributes' in data:
                network = np.eye(data['Attributes'].shape[0])
            elif 'X' in data:
                network = np.eye(data['X'].shape[0])
            elif 'features' in data:
                network = np.eye(data['features'].shape[0])
            elif 'feature' in data:
                network = np.eye(data['feature'].shape[0])
            elif 'node_features' in data:
                network = np.eye(data['node_features'].shape[0])
            else:
                network = np.eye(100)
            print(f"Warning:  {network.shape} ")


        if 'Attributes' in data:
            attr = data['Attributes']
        elif 'X' in data:
            attr = data['X']
        elif 'features' in data:
            attr = data['features']
        elif 'feature' in data:
            attr = data['feature']
        elif 'node_features' in data:
            attr = data['node_features']
        elif 'feat' in data:
            attr = data['feat']
        else:
            print("Warning: ,")
            attr = np.eye(network.shape[0])


        if 'Label' in data:
            label = data['Label']
        elif 'gnd' in data:
            label = data['gnd']
        elif 'label' in data:
            label = data['label']
        elif 'labels' in data:
            label = data['labels']
        else:
            print("Warning: ,0")
            label = np.zeros((network.shape[0], 1))


        train_idx = data.get('train_idx', None)
        val_idx = data.get('val_idx', None)
        test_idx = data.get('test_idx', None)
    elif actual_path.endswith('.npz'):

        npz_data = np.load(actual_path)



        data = {}
        for key in npz_data.keys():
            data[key] = npz_data[key]


        network = None
        for key in ['edges', 'edge_index', 'adj', 'adjacency', 'A', 'network']:
            if key in data:
                if key == 'edges' or key == 'edge_index':

                    edge_index = data[key]
                    if isinstance(edge_index, np.ndarray):
                        n = int(np.max(edge_index)) + 1
                        network = np.zeros((n, n))
                        if edge_index.shape[0] == 2 or edge_index.shape[1] == 2:

                            if edge_index.shape[0] == 2:
                                for i in range(edge_index.shape[1]):
                                    src, dst = edge_index[0, i], edge_index[1, i]
                                    network[src, dst] = 1
                                    network[dst, src] = 1
                            else:
                                for i in range(edge_index.shape[0]):
                                    src, dst = edge_index[i, 0], edge_index[i, 1]
                                    network[src, dst] = 1
                                    network[dst, src] = 1
                else:
                    network = data[key]
                break

        if network is None:

            for key in ['node_features', 'features', 'X', 'attr', 'feature']:
                if key in data:
                    network = np.eye(data[key].shape[0])
                    break


        attr = None
        for key in ['node_features', 'features', 'X', 'attr', 'attribute']:
            if key in data:
                attr = data[key]
                break


        label = None
        for key in ['node_labels', 'label', 'labels', 'Label', 'gnd', 'y']:
            if key in data:
                label = data[key]
                break


        train_idx = data.get('train_idx', None)
        val_idx = data.get('val_idx', None)
        test_idx = data.get('test_idx', None)
    else:
        raise ValueError(f": {actual_path}")


    if network is None or attr is None or label is None:
        raise ValueError(":,")


    adj = sp.csr_matrix(network)
    feat = sp.lil_matrix(attr)


    ano_labels = np.squeeze(np.array(label))

    if len(ano_labels.shape) > 1:

        if ano_labels.shape[0] == 1:
            ano_labels = ano_labels[0]

        else:
            ano_labels = ano_labels[:, 0]

    if isinstance(data, dict):
        if 'str_anomaly_label' in data:
            str_ano_labels = np.squeeze(np.array(data['str_anomaly_label']))

            if len(str_ano_labels.shape) > 1:
                if str_ano_labels.shape[0] == 1:
                    str_ano_labels = str_ano_labels[0]
                else:
                    str_ano_labels = str_ano_labels[:, 0]
        else:
            str_ano_labels = None

        if 'attr_anomaly_label' in data:
            attr_ano_labels = np.squeeze(np.array(data['attr_anomaly_label']))

            if len(attr_ano_labels.shape) > 1:
                if attr_ano_labels.shape[0] == 1:
                    attr_ano_labels = attr_ano_labels[0]
                else:
                    attr_ano_labels = attr_ano_labels[:, 0]
        else:
            attr_ano_labels = None
    else:
        str_ano_labels = None
        attr_ano_labels = None




    num_node = adj.shape[0]  # total nodes
    all_idx = list(range(num_node))

    normal_idx = [i for i in all_idx if ano_labels[i] == 0]
    abnormal_idx = [i for i in all_idx if ano_labels[i] == 1]
    random.shuffle(normal_idx)
    random.shuffle(abnormal_idx)

    num_train_normal = int(len(normal_idx) * train_rate)
    num_train_normal = min(num_train_normal, len(normal_idx))
    if train_rate > 0 and num_train_normal < 1 and len(normal_idx) > 0:
        num_train_normal = 1

    idx_train = normal_idx[:num_train_normal]
    idx_val = []
    idx_test = normal_idx[num_train_normal:] + abnormal_idx

    random.shuffle(idx_train)
    random.shuffle(idx_test)

    print(f"[normal] train_rate={train_rate:.2f} | "
          f"normal={len(idx_train)}({len(idx_train)/max(1, len(normal_idx))*100:.1f}% of normal nodes) | "
          f"=normal{len(normal_idx) - len(idx_train)} + {len(abnormal_idx)}")


    train_labels = np.squeeze(ano_labels[idx_train]).tolist()
    test_labels = np.squeeze(ano_labels[idx_test]).tolist()

    if isinstance(train_labels, list) and train_labels and isinstance(train_labels[0], list):
        train_labels = [item for sublist in train_labels for item in sublist]
        test_labels = [item for sublist in test_labels for item in sublist]




    all_normal_label_idx = [i for i in idx_train if ano_labels[i] == 0]
    rate = 0.5
    normal_label_idx = all_normal_label_idx[: int(
        len(all_normal_label_idx) * rate)]



    # contamination
    # real_abnormal_id = np.array(all_idx)[np.argwhere(ano_labels == 1).squeeze()].tolist()
    # add_rate = 0.1 * len(real_abnormal_id)
    # random.shuffle(real_abnormal_id)
    # add_abnormal_id = real_abnormal_id[:int(add_rate)]
    # normal_label_idx = normal_label_idx + add_abnormal_id
    # idx_test = np.setdiff1d(idx_test, add_abnormal_id, False)


    random.shuffle(normal_label_idx)

    if dataset in ['Amazon']:
        # abnormal_label_idx = normal_label_idx[: int(
            # len(normal_label_idx) * 0.05)]
        abnormal_label_idx = normal_label_idx[: int(
            len(normal_label_idx) * 0.15)]
    else:
        abnormal_label_idx = normal_label_idx[: int(
            len(normal_label_idx) * 0.15)]


    return adj, feat, ano_labels, all_idx, idx_train, idx_val, idx_test, ano_labels, str_ano_labels, attr_ano_labels, normal_label_idx, abnormal_label_idx


def adj_to_dgl_graph(adj):
    """Adj to dgl graph."""
    import dgl
    import networkx as nx
    print("  DGL...")
    try:
        nx_graph = nx.from_scipy_sparse_array(adj)
    except AttributeError:
        nx_graph = nx.from_scipy_sparse_matrix(adj)
    g = dgl.from_networkx(nx_graph)
    print(f"  DGL: {g.number_of_nodes()}, {g.number_of_edges()}")
    return g
# def adj_to_dgl_graph(adj):


#     start_time = time.time()

#     try:

#         nx_start = time.time()
#         try:
#             nx_graph = nx.from_scipy_sparse_array(adj)
#         except AttributeError:
#             try:
#                 nx_graph = nx.from_scipy_sparse_matrix(adj)
#             except:
#                 nx_graph = nx.Graph(adj)
#         nx_time = time.time() - nx_start



#         dgl_start = time.time()
#         g = dgl.from_networkx(nx_graph)
#         dgl_time = time.time() - dgl_start


#         total_time = time.time() - start_time



#         return g

#     except Exception as e:

#         raise


# def adj_to_dgl_graph(adj):

#     try:

#         nx_graph = nx.from_scipy_sparse_array(adj)
#     except AttributeError:

#         try:
#             nx_graph = nx.from_scipy_sparse_matrix(adj)
#         except:

#             nx_graph = nx.Graph(adj)


#     g = dgl.from_networkx(nx_graph)
#     return g



def generate_rwr_subgraph(dgl_graph, subgraph_size):
    """Generate rwr subgraph."""
    print("  :DGL,")
    return []





matplotlib.use('Agg')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['figure.figsize'] = (8.5, 7.5)


def draw_pdf(message_normal, message_abnormal, message_real_abnormal, dataset, epoch):
    """Draw pdf."""
    message_all = [np.squeeze(message_normal), np.squeeze(
        message_abnormal), np.squeeze(message_real_abnormal)]

    mu_0 = np.mean(message_all[0])
    sigma_0 = np.std(message_all[0])
    mu_1 = np.mean(message_all[1])  # synthetic anomaliesmean
    sigma_1 = np.std(message_all[1])
    mu_2 = np.mean(message_all[2])
    sigma_2 = np.std(message_all[2])


    n, bins, patches = plt.hist(
        message_all, bins=30, normed=1, label=['normal', 'synthetic anomalies', ''])

    y_0 = mlab.normpdf(bins, mu_0, sigma_0)
    y_1 = mlab.normpdf(bins, mu_1, sigma_1)
    y_2 = mlab.normpdf(bins, mu_2, sigma_2)

    plt.plot(bins, y_0, color='steelblue', linestyle='--', linewidth=7.5)
    plt.plot(bins, y_1, color='darkorange', linestyle='--', linewidth=7.5)
    plt.plot(bins, y_2, color='green', linestyle='--', linewidth=7.5)

    plt.ylim(0, 20)


    plt.yticks(fontsize=30)
    plt.xticks(fontsize=30)

    plt.savefig('fig/{}/{}_{}.pdf'.format(dataset, dataset, epoch))
    plt.close()


def draw_pdf_methods(method, message_normal, message_abnormal, message_real_abnormal, dataset, epoch):
    """Draw pdf methods."""
    message_all = [np.squeeze(message_normal), np.squeeze(
        message_abnormal), np.squeeze(message_real_abnormal)]
    mu_0 = np.mean(message_all[0])
    sigma_0 = np.std(message_all[0])
    mu_1 = np.mean(message_all[1])
    sigma_1 = np.std(message_all[1])
    mu_2 = np.mean(message_all[2])
    sigma_2 = np.std(message_all[2])

    n, bins, patches = plt.hist(
        message_all, bins=30, normed=1, label=['normal', 'synthetic anomalies', ''])
    y_0 = mlab.normpdf(bins, mu_0, sigma_0)
    y_1 = mlab.normpdf(bins, mu_1, sigma_1)
    y_2 = mlab.normpdf(bins, mu_2, sigma_2)
    plt.plot(bins, y_0, color='steelblue', linestyle='--', linewidth=7.5)
    plt.plot(bins, y_1, color='darkorange', linestyle='--', linewidth=7.5)
    plt.plot(bins, y_2, color='green', linestyle='--', linewidth=7.5)
    plt.ylim(0, 8)

    plt.yticks(fontsize=30)
    plt.xticks(fontsize=30)

    plt.savefig('fig/{}/{}2/{}_{}.svg'.format(method, dataset, dataset, epoch))
    plt.close()
# import numpy as np
# import networkx as nx
# import scipy.sparse as sp
# import torch
# import scipy.io as sio
# import random
# import dgl
# from collections import Counter


# def sparse_to_tuple(sparse_mx, insert_batch=False):
#     """Convert sparse matrix to tuple representation."""
#     """Set insert_batch=True if you want to insert a batch dimension."""

#     def to_tuple(mx):
#         if not sp.isspmatrix_coo(mx):
#             mx = mx.tocoo()
#         if insert_batch:
#             coords = np.vstack((np.zeros(mx.row.shape[0]), mx.row, mx.col)).transpose()
#             values = mx.data
#             shape = (1,) + mx.shape
#         else:
#             coords = np.vstack((mx.row, mx.col)).transpose()
#             values = mx.data
#             shape = mx.shape
#         return coords, values, shape

#     if isinstance(sparse_mx, list):
#         for i in range(len(sparse_mx)):
#             sparse_mx[i] = to_tuple(sparse_mx[i])
#     else:
#         sparse_mx = to_tuple(sparse_mx)

#     return sparse_mx


# def preprocess_features(features):
#     """Row-normalize feature matrix and convert to tuple representation"""
#     rowsum = np.array(features.sum(1))
#     r_inv = np.power(rowsum, -1).flatten()
#     r_inv[np.isinf(r_inv)] = 0.
#     r_mat_inv = sp.diags(r_inv)
#     features = r_mat_inv.dot(features)
#     return features.todense(), sparse_to_tuple(features)


# def normalize_adj(adj):
#     """Symmetrically normalize adjacency matrix."""
#     adj = sp.coo_matrix(adj)
#     rowsum = np.array(adj.sum(1))
#     d_inv_sqrt = np.power(rowsum, -0.5).flatten()
#     d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
#     d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
#     return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


# def dense_to_one_hot(labels_dense, num_classes):
#     """Convert class labels from scalars to one-hot vectors."""
#     num_labels = labels_dense.shape[0]
#     index_offset = np.arange(num_labels) * num_classes
#     labels_one_hot = np.zeros((num_labels, num_classes))
#     labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
#     return labels_one_hot


# def load_mat(dataset, train_rate=0.3, val_rate=0.1):

#     """Load .mat dataset."""
#     data = sio.loadmat("./dataset/{}.mat".format(dataset))
#     label = data['Label'] if ('Label' in data) else data['gnd']
#     attr = data['Attributes'] if ('Attributes' in data) else data['X']
#     network = data['Network'] if ('Network' in data) else data['A']

#     adj = sp.csr_matrix(network)
#     feat = sp.lil_matrix(attr)

#     # labels = np.squeeze(np.array(data['Class'], dtype=np.int64) - 1)
#     # num_classes = np.max(labels) + 1
#     # labels = dense_to_one_hot(labels, num_classes)

#     ano_labels = np.squeeze(np.array(label))
#     if 'str_anomaly_label' in data:
#         str_ano_labels = np.squeeze(np.array(data['str_anomaly_label']))
#         attr_ano_labels = np.squeeze(np.array(data['attr_anomaly_label']))
#     else:
#         str_ano_labels = None
#         attr_ano_labels = None

#     num_node = adj.shape[0]
#     num_train = int(num_node * train_rate)
#     num_val = int(num_node * val_rate)
#     all_idx = list(range(num_node))
#     random.shuffle(all_idx)
#     idx_train = all_idx[: num_train]
#     idx_val = all_idx[num_train: num_train + num_val]
#     idx_test = all_idx[num_train + num_val:]
#     # idx_test = all_idx[num_train:]
#     print('Training', Counter(np.squeeze(ano_labels[idx_train])))
#     print('Test', Counter(np.squeeze(ano_labels[idx_test])))
#     # Sample some labeled normal nodes
#     all_normal_label_idx = [i for i in idx_train if ano_labels[i] == 0]
#     rate = 0.5  #  change train_rate to 0.3 0.5 0.6  0.8
#     normal_label_idx = all_normal_label_idx[: int(len(all_normal_label_idx) * rate)]
#     print('Training rate', rate)

#     # normal_label_idx = all_normal_label_idx[: int(len(all_normal_label_idx) * 0.2)]
#     # normal_label_idx = all_normal_label_idx[: int(len(all_normal_label_idx) * 0.25)]
#     # normal_label_idx = all_normal_label_idx[: int(len(all_normal_label_idx) * 0.15)]
#     # normal_label_idx = all_normal_label_idx[: int(len(all_normal_label_idx) * 0.10)]

#     # contamination
#     # real_abnormal_id = np.array(all_idx)[np.argwhere(ano_labels == 1).squeeze()].tolist()
#     # add_rate = 0.1 * len(real_abnormal_id)
#     # random.shuffle(real_abnormal_id)
#     # add_abnormal_id = real_abnormal_id[:int(add_rate)]
#     # normal_label_idx = normal_label_idx + add_abnormal_id
#     # idx_test = np.setdiff1d(idx_test, add_abnormal_id, False)

#     # contamination
#     # real_abnormal_id = np.array(all_idx)[np.argwhere(ano_labels == 1).squeeze()].tolist()
#     # add_rate = 0.05 * len(real_abnormal_id)  #0.05 0.1  0.15
#     # remove_rate = 0.15 * len(real_abnormal_id)
#     # random.shuffle(real_abnormal_id)
#     # add_abnormal_id = real_abnormal_id[:int(add_rate)]
#     # remove_abnormal_id = real_abnormal_id[:int(remove_rate)]
#     # normal_label_idx = normal_label_idx + add_abnormal_id
#     # idx_test = np.setdiff1d(idx_test, remove_abnormal_id, False)

#     # camouflage
#     # real_abnormal_id = np.array(all_idx)[np.argwhere(ano_labels == 1).squeeze()].tolist()
#     # normal_feat = np.mean(feat[normal_label_idx], 0)
#     # replace_rate = 0.05 * normal_feat.shape[1]
#     # feat[real_abnormal_id, :int(replace_rate)] = normal_feat[:, :int(replace_rate)]

#     random.shuffle(normal_label_idx)
#     # 0.05 for Amazon and 0.15 for other datasets
#     if dataset in ['Amazon']:
#         abnormal_label_idx = normal_label_idx[: int(len(normal_label_idx) * 0.05)]
#     else:
#         abnormal_label_idx = normal_label_idx[: int(len(normal_label_idx) * 0.15)]
#     return adj, feat, ano_labels, all_idx, idx_train, idx_val, idx_test, ano_labels, str_ano_labels, attr_ano_labels, normal_label_idx, abnormal_label_idx


# # def adj_to_dgl_graph(adj):
# #     """Convert adjacency matrix to dgl format."""
# #     nx_graph = nx.from_scipy_sparse_matrix(adj)
# #     dgl_graph = dgl.DGLGraph(nx_graph)
# #     return dgl_graph

# # NetworkXfrom_scipy_sparse_matrix
# #

# def adj_to_dgl_graph(adj):
#     """Convert scipy sparse matrix to DGL graph"""
#     try:
#         # NetworkX
#         nx_graph = nx.from_scipy_sparse_array(adj)
#     except AttributeError:
#         #
#         try:
#             nx_graph = nx.from_scipy_sparse_matrix(adj)
#         except:
#             #
#             nx_graph = nx.Graph(adj)

#     # DGL
#     g = dgl.from_networkx(nx_graph)
#     return g

# def generate_rwr_subgraph(dgl_graph, subgraph_size):
#     """Generate subgraph with RWR algorithm."""
#     all_idx = list(range(dgl_graph.number_of_nodes()))
#     reduced_size = subgraph_size - 1
#     traces = dgl.contrib.sampling.random_walk_with_restart(dgl_graph, all_idx, restart_prob=1,
#                                                            max_nodes_per_seed=subgraph_size * 3)
#     subv = []

#     for i, trace in enumerate(traces):
#         subv.append(torch.unique(torch.cat(trace), sorted=False).tolist())
#         retry_time = 0
#         while len(subv[i]) < reduced_size:
#             cur_trace = dgl.contrib.sampling.random_walk_with_restart(dgl_graph, [i], restart_prob=0.9,
#                                                                       max_nodes_per_seed=subgraph_size * 5)
#             subv[i] = torch.unique(torch.cat(cur_trace[0]), sorted=False).tolist()
#             retry_time += 1
#             if (len(subv[i]) <= 2) and (retry_time > 10):
#                 subv[i] = (subv[i] * reduced_size)
#         subv[i] = subv[i][:reduced_size * 3]
#         subv[i].append(i)

#     return subv


# import matplotlib.pyplot as plt
# import matplotlib.mlab as mlab
# import matplotlib

# matplotlib.use('Agg')
# plt.rcParams['figure.dpi'] = 300  #
# plt.rcParams['figure.figsize'] = (8.5, 7.5)
# # plt.rcParams['figure.figsize'] = (10.5, 9.5)
# from matplotlib.backends.backend_pdf import PdfPages


# def draw_pdf(message_normal, message_abnormal, message_real_abnormal, dataset, epoch):
#     message_all = [np.squeeze(message_normal), np.squeeze(message_abnormal), np.squeeze(message_real_abnormal)]
#     mu_0 = np.mean(message_all[0])  #
#     sigma_0 = np.std(message_all[0])
#     # print('The mean of normal {}'.format(mu_0))
#     # print('The std of normal {}'.format(sigma_0))
#     mu_1 = np.mean(message_all[1])  #
#     sigma_1 = np.std(message_all[1])
#     # print('The mean of abnormal {}'.format(mu_1))
#     # print('The std of abnormal {}'.format(sigma_1))
#     mu_2 = np.mean(message_all[2])  #
#     sigma_2 = np.std(message_all[2])
#     # print('The mean of abnormal {}'.format(mu_2))
#     # print('The std of abnormal {}'.format(sigma_2))
#     n, bins, patches = plt.hist(message_all, bins=30, normed=1, label=['Normal', 'Outlier', 'Abnormal'])
#     y_0 = mlab.normpdf(bins, mu_0, sigma_0)  # y
#     y_1 = mlab.normpdf(bins, mu_1, sigma_1)  # y
#     y_2 = mlab.normpdf(bins, mu_2, sigma_2)  # y
#     # plt.plot(bins, y_0, 'g--', linewidth=3.5)  # y
#     # plt.plot(bins, y_1, 'r--', linewidth=3.5)  # y
#     plt.plot(bins, y_0, color='steelblue', linestyle='--', linewidth=7.5)  # y
#     plt.plot(bins, y_1, color='darkorange', linestyle='--', linewidth=7.5)  # y
#     plt.plot(bins, y_2, color='green', linestyle='--', linewidth=7.5)  # y
#     plt.ylim(0, 20)

#     # plt.xlabel('RAW-based Affinity', fontsize=25)
#     # plt.xlabel('TAM-based Affinity', fontsize=25)
#     # plt.ylabel('Number of Samples', size=25)
#     plt.yticks(fontsize=30)
#     plt.xticks(fontsize=30)
#     # from matplotlib.pyplot import MultipleLocator
#     # x_major_locator = MultipleLocator(0.02)
#     # ax = plt.gca()
#     # ax.xaxis.set_major_locator(x_major_locator)
#     # plt.legend(loc='upper left', fontsize=30)
#     # plt.title('Amazon'.format(dataset), fontsize=25)
#     # plt.title('BlogCatalog', fontsize=50)
#     plt.savefig('fig/{}/{}_{}.pdf'.format(dataset, dataset, epoch))
#     plt.close()


# def draw_pdf_methods(method, message_normal, message_abnormal, message_real_abnormal, dataset, epoch):
#     message_all = [np.squeeze(message_normal), np.squeeze(message_abnormal), np.squeeze(message_real_abnormal)]
#     mu_0 = np.mean(message_all[0])  #
#     sigma_0 = np.std(message_all[0])
#     # print('The mean of normal {}'.format(mu_0))
#     # print('The std of normal {}'.format(sigma_0))
#     mu_1 = np.mean(message_all[1])  #
#     sigma_1 = np.std(message_all[1])
#     # print('The mean of abnormal {}'.format(mu_1))
#     # print('The std of abnormal {}'.format(sigma_1))
#     mu_2 = np.mean(message_all[2])  #
#     sigma_2 = np.std(message_all[2])
#     # print('The mean of abnormal {}'.format(mu_2))
#     # print('The std of abnormal {}'.format(sigma_2))

#     n, bins, patches = plt.hist(message_all, bins=30, normed=1, label=['Normal', 'Outlier', 'Abnormal'])
#     y_0 = mlab.normpdf(bins, mu_0, sigma_0)  # y
#     y_1 = mlab.normpdf(bins, mu_1, sigma_1)  # y
#     y_2 = mlab.normpdf(bins, mu_2, sigma_2)  # y
#     # plt.plot(bins, y_0, 'g--', linewidth=3.5)  # y
#     # plt.plot(bins, y_1, 'r--', linewidth=3.5)  # y
#     plt.plot(bins, y_0, color='steelblue', linestyle='--', linewidth=7.5)  # y
#     plt.plot(bins, y_1, color='darkorange', linestyle='--', linewidth=7.5)  # y
#     plt.plot(bins, y_2, color='green', linestyle='--', linewidth=7.5)  # y
#     plt.ylim(0, 8)

#     # plt.xlabel('RAW-based Affinity', fontsize=25)
#     # plt.xlabel('TAM-based Affinity', fontsize=25)
#     # plt.ylabel('Number of Samples', size=25)

#     plt.yticks(fontsize=30)
#     plt.xticks(fontsize=30)
#     # plt.legend(loc='upper left', fontsize=30)
#     # plt.title('Amazon'.format(dataset), fontsize=25)
#     # plt.title('BlogCatalog', fontsize=50)
#     plt.savefig('fig/{}/{}2/{}_{}.svg'.format(method, dataset, dataset, epoch))
#     plt.close()



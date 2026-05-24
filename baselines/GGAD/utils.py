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
# 临时注释：移除DGL依赖，避免dll错误
# import dgl
from collections import Counter
import time
import os

# utils.py 中新增
import scipy.io as sio


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """将scipy稀疏矩阵转换为torch稀疏张量"""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)


def is_sparse_matrix(matrix):
    """检查矩阵是否为稀疏格式"""
    return hasattr(matrix, 'tocoo') or (isinstance(matrix, torch.Tensor) and matrix.is_sparse)


def save_fake_anomaly_dataset(adj, features, normal_idx, fake_anomaly_idx, dataset_name, save_path):
    """
    保存仅包含正常节点和伪造异常节点的数据集（不含真实异常节点）
    adj: 原始邻接矩阵（稀疏矩阵）
    features: 原始节点特征（稀疏/稠密矩阵）
    normal_idx: 正常节点索引（list）
    fake_anomaly_idx: 伪造异常节点索引（list）
    dataset_name: 数据集名称
    save_path: 保存路径（如'fake_datasets/'）
    """
    # 合并正常节点和伪造异常节点的索引（确保无重复）
    all_nodes = list(set(normal_idx + fake_anomaly_idx))
    all_nodes.sort()  # 按索引排序
    n = len(all_nodes)

    # 构建子图邻接矩阵（仅保留all_nodes之间的边）
    adj_sub = adj[all_nodes, :][:, all_nodes]  # 子图邻接矩阵
    # 构建子图特征矩阵
    features_sub = features[all_nodes]  # 子图特征

    # 构建标签：0=正常节点，1=伪造异常节点
    labels_sub = np.zeros(n, dtype=int)
    # 映射原始索引到子图中的新索引（因为all_nodes是排序后的）
    idx_map = {orig_idx: new_idx for new_idx, orig_idx in enumerate(all_nodes)}
    for orig_idx in fake_anomaly_idx:
        new_idx = idx_map[orig_idx]
        labels_sub[new_idx] = 1  # 伪造异常节点标签为1

    # 创建保存目录
    os.makedirs(save_path, exist_ok=True)
    save_file = f"{save_path}/{dataset_name}_fake_anomalies.mat"

    # 保存为.mat文件（方便后续加载和跳数分析）
    sio.savemat(
        save_file,
        {
            'Network': adj_sub,  # 子图邻接矩阵
            'Attributes': features_sub,  # 子图特征
            'Label': labels_sub.reshape(1, -1),  # 标签（1×n）
            'original_indices': np.array(all_nodes),  # 原始节点索引（用于追溯）
            # 伪造节点的原始索引
            'fake_anomaly_original_indices': np.array(fake_anomaly_idx)
        }
    )
    print(f"\n[伪造节点数据集保存] 已保存到: {save_file}")
    print(
        f"[保存信息] 总节点数: {n} (正常: {len(normal_idx)}, 伪造异常: {len(fake_anomaly_idx)})")
    print(f"[保存信息] 不含真实异常节点，可直接用于跳数分析\n")


def sparse_to_tuple(sparse_mx, insert_batch=False):
    """把稀疏矩阵转成 tuple 格式（坐标+值+形状）"""
    """如果插了 batch 维度，就多塞个0在前面"""

    def to_tuple(mx):
        # 先转成 COO 格式（稀疏矩阵的一种存储方式，存坐标和值）
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        if insert_batch:
            # 加个 batch 维度，坐标前面多一行0
            coords = np.vstack(
                (np.zeros(mx.row.shape[0]), mx.row, mx.col)).transpose()
            values = mx.data
            shape = (1,) + mx.shape  # 形状也加个1在前面
        else:
            # 正常情况，坐标就是（行，列）
            coords = np.vstack((mx.row, mx.col)).transpose()
            values = mx.data
            shape = mx.shape
        return coords, values, shape

    # 如果是列表，就逐个转；不然直接转单个
    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """把特征矩阵做行归一化，再转成 tuple 格式"""
    import scipy.sparse as sp
    
    # 确保是稀疏矩阵
    if not sp.issparse(features):
        features = sp.csr_matrix(features)
    
    # 算每行的和（每个节点的特征总和）
    rowsum = np.array(features.sum(1))
    # 求倒数（用来归一化，让每行和为 1）
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.  # 避免除 0 出错
    
    # 修复：对于全零特征的节点，添加微小扰动避免数值不稳定
    zero_rows = np.where(rowsum == 0)[0]
    if len(zero_rows) > 0:
        print(f"[预处里] 发现 {len(zero_rows)} 个节点特征全为 0，添加微小扰动")
        # 给全零行添加微小随机值
        features = features.tocsr()
        for idx in zero_rows:
            # 在该行随机位置添加小的非零值
            features[idx, np.random.randint(0, features.shape[1])] = 1e-6
        features = features.tocoo()
        # 重新计算 rowsum
        rowsum = np.array(features.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
    
    # 建一个对角矩阵，对角线是上面的倒数
    r_mat_inv = sp.diags(r_inv)
    # 用对角矩阵左乘特征矩阵，实现行归一化
    features = r_mat_inv.dot(features)
    return features.todense(), sparse_to_tuple(features)


def normalize_adj(adj):
    """对称归一化邻接矩阵（图神经网络的常规操作，让计算稳定）"""
    adj = sp.coo_matrix(adj)
    # 算每行的和（每个节点的度）
    rowsum = np.array(adj.sum(1))
    # 度的开方的倒数（对称归一化公式里要用）
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.  # 孤立节点的度为0，这里设为0
    # 建对角矩阵
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    # 对称归一化公式：D^(-1/2) * A * D^(-1/2)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def dense_to_one_hot(labels_dense, num_classes):
    """把普通标签转成 one-hot 格式（比如把[0,1,2]转成[[1,0,0], [0,1,0], [0,0,1]]）"""
    num_labels = labels_dense.shape[0]
    # 算每个标签在 one-hot 里的位置
    index_offset = np.arange(num_labels) * num_classes
    # 初始化全0矩阵
    labels_one_hot = np.zeros((num_labels, num_classes))
    # 对应位置设为1
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def load_mat(dataset, train_rate=0.3, val_rate=0.1, file_path=None):
    """加载 .mat 或 .pkl 格式的数据集
    dataset: 数据集名称或完整文件路径
    train_rate: 训练集比例
    val_rate: 验证集比例
    file_path: 完整文件路径（如果提供，将忽略dataset参数的文件名部分）
    """
    import pickle
    
    # 确定实际文件路径
    if file_path is None:
        # 根据文件扩展名自动选择加载方式
        if dataset.endswith('.pkl'):
            actual_path = dataset
        elif dataset.endswith('.mat'):
            actual_path = dataset
        else:
            # 支持多路径优先级
            # 获取当前文件的目录，然后向上一级找到datasets目录
            utils_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(utils_dir)
            datasets_dir = os.path.join(project_root, 'datasets')
            
            dataset_root = os.environ.get("DATASET_DIR", "/root/dataset")
            possible_paths = [
                os.path.join(dataset_root, f"{dataset}.mat"),
                os.path.join(dataset_root, f"{dataset}.npz"),
                os.path.join(datasets_dir, f"{dataset}.mat"),
                os.path.join(datasets_dir, f"{dataset}.npz"),
                f"../datasets/{dataset}.mat",
                f"../datasets/{dataset}.npz",
                f"../../datasets/{dataset}.mat",
                f"../../datasets/{dataset}.npz",
                f"../../../datasets/{dataset}.mat",
                f"../../../datasets/{dataset}.npz",
                f"C:\\Users\\27586\\Desktop\\Sync_Island\\datasets\\{dataset}.mat",
                f"C:\\Users\\27586\\Desktop\\Sync_Island\\datasets\\{dataset}.npz",
            ]
            actual_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    actual_path = path
                    break
            if actual_path is None:
                # 默认使用第一个路径
                actual_path = possible_paths[0]
    else:
        actual_path = file_path
    
    # 根据文件扩展名选择加载方法
    if actual_path.endswith('.pkl'):
        # 加载 pkl 格式文件
        with open(actual_path, 'rb') as f:
            data = pickle.load(f)
        
        # pkl 文件的常见数据结构处理
        if isinstance(data, tuple) and len(data) >= 3:
            # 假设格式为 (adj, feat, label)
            adj, feat, label = data
            network = adj
            attr = feat
        elif isinstance(data, dict):
            # 取标签（不同数据集可能字段名不一样）
            label = data['Label'] if ('Label' in data) else data['gnd'] if ('gnd' in data) else data.get('label', None)
            # 取节点特征
            attr = data['Attributes'] if ('Attributes' in data) else data['X'] if ('X' in data) else data.get('feat', None) if ('feat' in data) else data.get('features', None) if ('features' in data) else data.get('feature', None)
            # 取邻接矩阵
            network = data['Network'] if ('Network' in data) else data['A'] if ('A' in data) else data.get('adj', None) if ('adj' in data) else data.get('homo', None)
        else:
            raise ValueError(f"不支持的 pkl 数据格式: {type(data)}")
    elif actual_path.endswith('.mat'):
        # 加载 mat 格式文件
        data = sio.loadmat(actual_path)
        # print(f"成功加载 .mat 文件，包含的键: {list(data.keys())}")
        
        # 处理特殊数据集格式
        if 'PAP' in data and 'PLP' in data and 'PTP' in data:
            # ACM数据集：多层网络，选择PAP作为主要邻接矩阵（论文-作者-论文）
            network = data['PAP']
        elif 'homo' in data:
            # YelpChi数据集：使用同构图表示
            network = data['homo']
        elif 'adj_matrix' in data:
            # graph_fraud_dataset：使用adj_matrix作为邻接矩阵
            network = data['adj_matrix']
        elif 'net_APTPA' in data or 'net_APCPA' in data or 'net_APA' in data:
            # DBLP数据集：使用net_APA（作者-论文-作者）作为主要邻接矩阵
            network = data['net_APA']
        elif 'MAM' in data or 'MDM' in data or 'MYM' in data:
            # IMDB5K数据集：使用MAM（电影-演员-电影）作为主要邻接矩阵
            network = data['MAM']
        elif 'Network' in data:
            network = data['Network']
        elif 'A' in data:
            network = data['A']
        elif 'adj' in data:
            network = data['adj']
        else:
            print("警告: 未找到邻接矩阵，尝试从其他键生成")
            # 尝试从其他可能的键生成邻接矩阵
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
            print(f"警告: 生成了一个 {network.shape} 的单位矩阵作为邻接矩阵")
        
        # 加载特征
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
            print("警告: 未找到特征数据，生成单位矩阵作为特征")
            attr = np.eye(network.shape[0])
        
        # 加载标签
        if 'Label' in data:
            label = data['Label']
        elif 'gnd' in data:
            label = data['gnd']
        elif 'label' in data:
            label = data['label']
        elif 'labels' in data:
            label = data['labels']
        else:
            print("警告: 未找到标签数据，生成全0标签")
            label = np.zeros((network.shape[0], 1))
        
        # 加载预定义的索引（如果有）
        train_idx = data.get('train_idx', None)
        val_idx = data.get('val_idx', None)
        test_idx = data.get('test_idx', None)
    elif actual_path.endswith('.npz'):
        # 加载 npz 格式文件
        npz_data = np.load(actual_path)
        # print(f"成功加载 .npz 文件，包含的键: {list(npz_data.keys())}")
        
        # 将npz数据转换为字典格式
        data = {}
        for key in npz_data.keys():
            data[key] = npz_data[key]
        
        # 尝试确定邻接矩阵
        network = None
        for key in ['edges', 'edge_index', 'adj', 'adjacency', 'A', 'network']:
            if key in data:
                if key == 'edges' or key == 'edge_index':
                    # 将边索引转换为邻接矩阵
                    edge_index = data[key]
                    if isinstance(edge_index, np.ndarray):
                        n = int(np.max(edge_index)) + 1
                        network = np.zeros((n, n))
                        if edge_index.shape[0] == 2 or edge_index.shape[1] == 2:
                            # 边索引格式 [2, num_edges] 或 [num_edges, 2]
                            if edge_index.shape[0] == 2:
                                for i in range(edge_index.shape[1]):
                                    src, dst = edge_index[0, i], edge_index[1, i]
                                    network[src, dst] = 1
                                    network[dst, src] = 1  # 无向图
                            else:
                                for i in range(edge_index.shape[0]):
                                    src, dst = edge_index[i, 0], edge_index[i, 1]
                                    network[src, dst] = 1
                                    network[dst, src] = 1  # 无向图
                else:
                    network = data[key]
                break
        
        if network is None:
            # 尝试从特征维度生成单位矩阵
            for key in ['node_features', 'features', 'X', 'attr', 'feature']:
                if key in data:
                    network = np.eye(data[key].shape[0])
                    break
        
        # 尝试确定特征
        attr = None
        for key in ['node_features', 'features', 'X', 'attr', 'attribute']:
            if key in data:
                attr = data[key]
                break
        
        # 尝试确定标签
        label = None
        for key in ['node_labels', 'label', 'labels', 'Label', 'gnd', 'y']:
            if key in data:
                label = data[key]
                break
        
        # 加载预定义的索引（如果有）
        train_idx = data.get('train_idx', None)
        val_idx = data.get('val_idx', None)
        test_idx = data.get('test_idx', None)
    else:
        raise ValueError(f"不支持的文件格式: {actual_path}")

    # 确保数据有效
    if network is None or attr is None or label is None:
        raise ValueError("数据加载失败：缺少邻接矩阵、特征或标签")

    # 转成稀疏矩阵格式（省内存）
    adj = sp.csr_matrix(network)
    feat = sp.lil_matrix(attr)

    # 把标签转成一维数组（0是正常，1是异常）
    ano_labels = np.squeeze(np.array(label))
    # 确保是一维数组
    if len(ano_labels.shape) > 1:
        # 处理 (1, N) 形状的标签
        if ano_labels.shape[0] == 1:
            ano_labels = ano_labels[0]
        # 处理 (N, 1) 形状的标签
        else:
            ano_labels = ano_labels[:, 0]
    # 有些数据集还分结构异常和属性异常，没有就设为 None
    if isinstance(data, dict):
        if 'str_anomaly_label' in data:
            str_ano_labels = np.squeeze(np.array(data['str_anomaly_label']))
            # 确保是一维数组
            if len(str_ano_labels.shape) > 1:
                if str_ano_labels.shape[0] == 1:
                    str_ano_labels = str_ano_labels[0]
                else:
                    str_ano_labels = str_ano_labels[:, 0]
        else:
            str_ano_labels = None
        
        if 'attr_anomaly_label' in data:
            attr_ano_labels = np.squeeze(np.array(data['attr_anomaly_label']))
            # 确保是一维数组
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

    # 改进的数据集划分逻辑 - 确保每个集合都包含异常样本
    num_node = adj.shape[0]  # 总节点数
    num_train = int(num_node * train_rate)  # 训练集大小：总节点 * 训练比例
    num_val = int(num_node * val_rate)      # 验证集大小：总节点 * 验证比例
    all_idx = list(range(num_node))  # 所有节点的索引
    
    # 分离正常和异常节点
    normal_idx = [i for i in range(num_node) if ano_labels[i] == 0]
    abnormal_idx = [i for i in range(num_node) if ano_labels[i] == 1]
    
    # 确保异常节点列表不为空
    if len(abnormal_idx) == 0:
        # 如果没有异常节点，使用原始随机划分
        random.shuffle(all_idx)
        idx_train = all_idx[: num_train]
        idx_val = all_idx[num_train: num_train + num_val]
        idx_test = all_idx[num_train + num_val:]
    else:
        # 确保异常节点在各个集合中都有分布
        random.shuffle(normal_idx)
        random.shuffle(abnormal_idx)
        
        # 计算每个集合中应该包含的异常节点数量
        num_abnormal = len(abnormal_idx)
        num_train_abnormal = max(1, int(num_abnormal * train_rate))
        num_val_abnormal = max(1, int(num_abnormal * val_rate))
        num_test_abnormal = max(1, num_abnormal - num_train_abnormal - num_val_abnormal)
        
        # 计算每个集合中应该包含的正常节点数量
        num_train_normal = num_train - num_train_abnormal
        num_val_normal = num_val - num_val_abnormal
        num_test_normal = num_node - num_train - num_val - num_test_abnormal
        
        # 确保正常节点数量不为负数
        num_train_normal = max(0, num_train_normal)
        num_val_normal = max(0, num_val_normal)
        num_test_normal = max(0, num_test_normal)
        
        # 构建各个集合
        idx_train = normal_idx[:num_train_normal] + abnormal_idx[:num_train_abnormal]
        idx_val = normal_idx[num_train_normal:num_train_normal+num_val_normal] + abnormal_idx[num_train_abnormal:num_train_abnormal+num_val_abnormal]
        idx_test = normal_idx[num_train_normal+num_val_normal:] + abnormal_idx[num_train_abnormal+num_val_abnormal:]
        
        # 打乱每个集合
        random.shuffle(idx_train)
        random.shuffle(idx_val)
        random.shuffle(idx_test)

    # 打印训练集和测试集里的异常/正常节点数量（Counter 是计数用的）
    train_labels = np.squeeze(ano_labels[idx_train]).tolist()
    test_labels = np.squeeze(ano_labels[idx_test]).tolist()
    # 确保是一维列表
    if isinstance(train_labels, list) and train_labels and isinstance(train_labels[0], list):
        train_labels = [item for sublist in train_labels for item in sublist]
        test_labels = [item for sublist in test_labels for item in sublist]
    # print('训练集里的正常/异常数量：', Counter(train_labels))
    # print('测试集里的正常/异常数量：', Counter(test_labels))

    # 挑一部分训练集里的正常节点当"已知正常样本"
    all_normal_label_idx = [i for i in idx_train if ano_labels[i] == 0]
    rate = 0.5  # 从正常节点里再挑一半当"明确的正常样本"（可以改）
    normal_label_idx = all_normal_label_idx[: int(
        len(all_normal_label_idx) * rate)]
    # print('训练时用的正常样本比例：', rate)

    # 下面这些注释掉的是一些数据增强方法（污染、伪装等，暂时不用管）
    # contamination
    # real_abnormal_id = np.array(all_idx)[np.argwhere(ano_labels == 1).squeeze()].tolist()
    # add_rate = 0.1 * len(real_abnormal_id)
    # random.shuffle(real_abnormal_id)
    # add_abnormal_id = real_abnormal_id[:int(add_rate)]
    # normal_label_idx = normal_label_idx + add_abnormal_id
    # idx_test = np.setdiff1d(idx_test, add_abnormal_id, False)

    # 打乱正常节点索引，从中挑一部分当"伪造异常节点"
    random.shuffle(normal_label_idx)
    # Amazon 数据集挑5%当伪造异常，其他数据集挑15%
    if dataset in ['Amazon']:
        abnormal_label_idx = normal_label_idx[: int(
            len(normal_label_idx) * 0.05)]
    else:
        abnormal_label_idx = normal_label_idx[: int(
            len(normal_label_idx) * 0.15)]

    # 返回各种处理好的数据
    return adj, feat, ano_labels, all_idx, idx_train, idx_val, idx_test, ano_labels, str_ano_labels, attr_ano_labels, normal_label_idx, abnormal_label_idx


def adj_to_dgl_graph(adj):
    """快速构建DGL图"""
    import dgl
    import networkx as nx
    print("  构建DGL图...")
    try:
        nx_graph = nx.from_scipy_sparse_array(adj)
    except AttributeError:
        nx_graph = nx.from_scipy_sparse_matrix(adj)
    g = dgl.from_networkx(nx_graph)
    print(f"  DGL图: {g.number_of_nodes()}节点, {g.number_of_edges()}边")
    return g
# def adj_to_dgl_graph(adj):
#     """把邻接矩阵转成 DGL 图格式"""
#     print("  开始构建DGL图...")
#     start_time = time.time()

#     try:
#         print("  步骤1: 转换为NetworkX图...")
#         nx_start = time.time()
#         try:
#             nx_graph = nx.from_scipy_sparse_array(adj)
#         except AttributeError:
#             try:
#                 nx_graph = nx.from_scipy_sparse_matrix(adj)
#             except:
#                 nx_graph = nx.Graph(adj)
#         nx_time = time.time() - nx_start
#         print(f"  NetworkX图构建完成，耗时: {nx_time:.2f}秒")

#         print("  步骤2: 转换为DGL图...")
#         dgl_start = time.time()
#         g = dgl.from_networkx(nx_graph)
#         dgl_time = time.time() - dgl_start
#         print(f"  DGL图构建完成，耗时: {dgl_time:.2f}秒")

#         total_time = time.time() - start_time
#         print(f"  DGL图构建总耗时: {total_time:.2f}秒")
#         print(f"  图信息: {g.number_of_nodes()} 节点, {g.number_of_edges()} 边")

#         return g

#     except Exception as e:
#         print(f"  DGL图构建失败: {e}")
#         raise


# def adj_to_dgl_graph(adj):
#     """把邻接矩阵转成 DGL 图格式（DGL是处理图的工具库）"""
#     try:
#         # 先用 NetworkX 转，新版本用 from_scipy_sparse_array
#         nx_graph = nx.from_scipy_sparse_array(adj)
#     except AttributeError:
#         # 老版本用 from_scipy_sparse_matrix
#         try:
#             nx_graph = nx.from_scipy_sparse_matrix(adj)
#         except:
#             # 实在不行就直接建图
#             nx_graph = nx.Graph(adj)

#     # 转成 DGL 图
#     g = dgl.from_networkx(nx_graph)
#     return g


# 临时注释：移除DGL依赖，避免dll错误
def generate_rwr_subgraph(dgl_graph, subgraph_size):
    """用 RWR 算法（随机游走带重启）生成子图（就是从每个节点出发，走几步取周围节点）"""
    print("  临时注释：DGL依赖已移除，返回空列表")
    return []


# 下面是画图相关的函数，用来画正常/异常节点的分布直方图

# 设置画图参数（分辨率、大小等）
matplotlib.use('Agg')  # 非交互式模式，直接存图
plt.rcParams['figure.dpi'] = 300  # 分辨率
plt.rcParams['figure.figsize'] = (8.5, 7.5)  # 图大小


def draw_pdf(message_normal, message_abnormal, message_real_abnormal, dataset, epoch):
    """画正常、伪造异常、真实异常的分布直方图"""
    # 把三类数据放一起
    message_all = [np.squeeze(message_normal), np.squeeze(
        message_abnormal), np.squeeze(message_real_abnormal)]
    # 算均值和标准差（用来画正态分布曲线）
    mu_0 = np.mean(message_all[0])  # 正常节点均值
    sigma_0 = np.std(message_all[0])  # 正常节点标准差
    mu_1 = np.mean(message_all[1])  # 伪造异常均值
    sigma_1 = np.std(message_all[1])  # 伪造异常标准差
    mu_2 = np.mean(message_all[2])  # 真实异常均值
    sigma_2 = np.std(message_all[2])  # 真实异常标准差

    # 画直方图
    n, bins, patches = plt.hist(
        message_all, bins=30, normed=1, label=['正常', '伪造异常', '真实异常'])
    # 画正态分布曲线
    y_0 = mlab.normpdf(bins, mu_0, sigma_0)  # 正常节点的理论分布
    y_1 = mlab.normpdf(bins, mu_1, sigma_1)  # 伪造异常的理论分布
    y_2 = mlab.normpdf(bins, mu_2, sigma_2)  # 真实异常的理论分布
    # 曲线样式（颜色、线型、线宽）
    plt.plot(bins, y_0, color='steelblue', linestyle='--', linewidth=7.5)
    plt.plot(bins, y_1, color='darkorange', linestyle='--', linewidth=7.5)
    plt.plot(bins, y_2, color='green', linestyle='--', linewidth=7.5)
    # y轴范围
    plt.ylim(0, 20)

    # 坐标轴字体大小
    plt.yticks(fontsize=30)
    plt.xticks(fontsize=30)
    # 存图（路径：fig/数据集名/数据集名_轮次.pdf）
    plt.savefig('fig/{}/{}_{}.pdf'.format(dataset, dataset, epoch))
    plt.close()  # 关图，避免重叠


def draw_pdf_methods(method, message_normal, message_abnormal, message_real_abnormal, dataset, epoch):
    """和上面类似，不过是按方法存图"""
    message_all = [np.squeeze(message_normal), np.squeeze(
        message_abnormal), np.squeeze(message_real_abnormal)]
    mu_0 = np.mean(message_all[0])
    sigma_0 = np.std(message_all[0])
    mu_1 = np.mean(message_all[1])
    sigma_1 = np.std(message_all[1])
    mu_2 = np.mean(message_all[2])
    sigma_2 = np.std(message_all[2])

    n, bins, patches = plt.hist(
        message_all, bins=30, normed=1, label=['正常', '伪造异常', '真实异常'])
    y_0 = mlab.normpdf(bins, mu_0, sigma_0)
    y_1 = mlab.normpdf(bins, mu_1, sigma_1)
    y_2 = mlab.normpdf(bins, mu_2, sigma_2)
    plt.plot(bins, y_0, color='steelblue', linestyle='--', linewidth=7.5)
    plt.plot(bins, y_1, color='darkorange', linestyle='--', linewidth=7.5)
    plt.plot(bins, y_2, color='green', linestyle='--', linewidth=7.5)
    plt.ylim(0, 8)  # 另一个y轴范围

    plt.yticks(fontsize=30)
    plt.xticks(fontsize=30)
    # 存图路径按方法分
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

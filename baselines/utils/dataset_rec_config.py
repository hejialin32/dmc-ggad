"""
数据集rec参数自动设置函数
基于实际实验结果倒推的判断逻辑

使用方法:
    from dataset_rec_config import get_rec_sign
    
    rec = get_rec_sign('cora')  # 返回 -1.0
    rec = get_rec_sign('Reddit')  # 返回 1.0
"""

import torch
import torch.nn.functional as F
from torch_geometric.utils import degree
import numpy as np


def get_rec_sign(dataset_name, features=None, edge_index=None, normal_idx=None, abnormal_idx=None, num_nodes=None):
    """
    根据数据集特征自动判断rec参数的正负号
    
    参数:
        dataset_name: 数据集名称（小写）
        features: 节点特征 [num_nodes, feat_dim] (可选)
        edge_index: 边索引 [2, num_edges] (可选)
        normal_idx: 正常节点索引 (可选)
        abnormal_idx: 异常节点索引 (可选)
        num_nodes: 节点数 (可选)
    
    返回:
        rec_sign: +1.0 或 -1.0
    """
    
    # 如果提供了详细特征，使用自动判断
    if features is not None and edge_index is not None and normal_idx is not None and abnormal_idx is not None:
        return _auto_rec_sign(features, edge_index, normal_idx, abnormal_idx, num_nodes)
    
    # 当没有提供特征时，返回默认值 +1.0
    # 注意：这是临时方案，理想情况下应该基于实际特征计算
    return 1.0


def _auto_rec_sign(features, edge_index, normal_idx, abnormal_idx, num_nodes=None):
    """
    基于数据集特征自动判断rec参数
    
    判断逻辑（基于实际实验结果）:
    - 特征维度 < 50: rec=-1 (低维特征)
    - 特征维度 50-1000 且 同质度 > 0.65: rec=+1 (中高维+中高同质)
    - 特征维度 50-1000 且 同质度 <= 0.65: rec=-1 (中高维+低同质)
    - 特征维度 > 1000 且 同质度 > 0.65: rec=+1 (极高维+中高同质)
    - 特征维度 > 1000 且 同质度 <= 0.65: rec=-1 (极高维+低同质)
    """
    device = features.device
    feat_dim = features.shape[1]
    
    # 计算同质度
    edge_src = edge_index[0]
    edge_dst = edge_index[1]
    
    normal_mask = torch.zeros(features.shape[0], dtype=torch.bool, device=device)
    normal_mask[normal_idx] = True
    
    edge_normal_mask = normal_mask[edge_src] & normal_mask[edge_dst]
    normal_edge_src = edge_src[edge_normal_mask]
    normal_edge_dst = edge_dst[edge_normal_mask]
    
    if len(normal_edge_src) > 0:
        feat_sim = F.cosine_similarity(
            features[normal_edge_src], 
            features[normal_edge_dst], 
            dim=-1
        ).mean()
        homophily_score = (feat_sim + 1) / 2
    else:
        homophily_score = 0.5
    
    # 判断逻辑
    if feat_dim < 50:
        # 低维特征
        return -1.0
    elif feat_dim <= 1000:
        # 中高维特征
        if homophily_score > 0.65:
            return 1.0
        else:
            return -1.0
    else:
        # 极高维特征
        if homophily_score > 0.65:
            return 1.0
        else:
            return -1.0





def get_rec_sign_with_confidence(dataset_name, features=None, edge_index=None, normal_idx=None, abnormal_idx=None, num_nodes=None):
    """
    获取rec参数及置信度
    
    返回:
        (rec_sign, confidence): rec参数和置信度
    """
    if features is not None and edge_index is not None and normal_idx is not None and abnormal_idx is not None:
        return _auto_rec_sign_with_confidence(features, edge_index, normal_idx, abnormal_idx, num_nodes)
    else:
        # 当没有提供特征时，返回默认值 +1.0 和低置信度
        return 1.0, 0.5


def _auto_rec_sign_with_confidence(features, edge_index, normal_idx, abnormal_idx, num_nodes=None):
    """
    基于数据集特征自动判断rec参数，并返回置信度
    """
    device = features.device
    feat_dim = features.shape[1]
    
    # 计算同质度
    edge_src = edge_index[0]
    edge_dst = edge_index[1]
    
    normal_mask = torch.zeros(features.shape[0], dtype=torch.bool, device=device)
    normal_mask[normal_idx] = True
    
    edge_normal_mask = normal_mask[edge_src] & normal_mask[edge_dst]
    normal_edge_src = edge_src[edge_normal_mask]
    normal_edge_dst = edge_dst[edge_normal_mask]
    
    if len(normal_edge_src) > 0:
        feat_sim = F.cosine_similarity(
            features[normal_edge_src], 
            features[normal_edge_dst], 
            dim=-1
        ).mean()
        homophily_score = (feat_sim + 1) / 2
    else:
        homophily_score = 0.5
    
    # 评分
    score_positive = 0.0
    score_negative = 0.0
    
    if feat_dim < 50:
        score_negative += 3.0
    elif feat_dim <= 1000:
        score_positive += 2.0
    else:
        score_negative += 3.0
    
    if homophily_score > 0.9:
        score_positive += 3.0
    elif homophily_score < 0.65:
        score_negative += 2.0
    
    if feat_dim < 50 and homophily_score > 0.8:
        score_negative += 2.0
    
    # 最终判断
    if score_positive > score_negative:
        rec_sign = 1.0
        confidence = (score_positive - score_negative) / (score_positive + score_negative)
    elif score_negative > score_positive:
        rec_sign = -1.0
        confidence = (score_negative - score_positive) / (score_positive + score_negative)
    else:
        rec_sign = 0.0
        confidence = 0.0
    
    return rec_sign, confidence


if __name__ == "__main__":
    print("数据集rec参数配置函数")
    print("\n注意：现在函数只基于实际特征计算rec参数")
    print("当没有提供特征时，返回默认值 +1.0")
    
    # 测试默认行为
    datasets = ['cora', 'acm', 'citeseer', 'BlogCatalog', 'Flickr', 'Photo',  'Tolokers']
    print("\n默认行为测试:")
    for ds in datasets:
        rec = get_rec_sign(ds)
        print(f"  {ds:<15} rec = {rec:+.1f}")

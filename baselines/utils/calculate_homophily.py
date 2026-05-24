import torch
import torch.nn.functional as F
from torch_geometric.utils import k_hop_subgraph
import numpy as np
import matplotlib.pyplot as plt
import os
from utils.utils import load_mat


def get_adaptive_regional_rec(features, edge_index, num_nodes, k=2, batch_size=500, max_edges=10000):
    """
    计算基于 K 阶局部微环境的自适应重构参数 (rec)
    返回: 一个形状为 [num_nodes] 的张量，包含每个节点专属的连续 rec 参数，范围 [-1, 1]
    """
    device = features.device
    # 初始化一个空张量，先用来装每个节点的原始 H_region
    raw_h_tensor = torch.zeros(num_nodes, device=device)
    
    # 批量处理节点，减少内存使用
    for i in range(0, num_nodes, batch_size):
        end = min(i + batch_size, num_nodes)
        # 计算处理进度百分比
        progress = (end / num_nodes) * 100
        print(f"  处理进度: {progress:.1f}%")
        
        # 遍历当前批次的每个节点
        for node_idx in range(i, end):
            # 第一步：极其优雅地抠出微环境（K 阶子图）
            subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph(
                node_idx=node_idx,
                num_hops=k,
                edge_index=edge_index,
                relabel_nodes=False, # 保持原图的节点编号
                num_nodes=num_nodes
            )
            
            # 第二步：计算这个微环境的"整体氛围" (区域同质性 H)
            if sub_edge_index.size(1) > 0:
                src, dst = sub_edge_index
                
                # 限制边的数量，避免内存不足
                if sub_edge_index.size(1) > max_edges:
                    # 随机采样一部分边
                    indices = torch.randperm(sub_edge_index.size(1))[:max_edges]
                    src = src[indices]
                    dst = dst[indices]
                
                # 计算微环境内所有边的余弦相似度 (范围是 [-1, 1])
                edge_sim = F.cosine_similarity(features[src], features[dst], dim=-1)
                # 将余弦相似度均值归一化到 [0, 1] 区间，这就是原始的 H_region
                H_region = (edge_sim.mean() + 1.0) / 2.0
            else:
                # 如果这是一个孤立节点（低密度到极点），给个中立值 0.5
                H_region = torch.tensor(0.5, device=device)
                
            # 此时不再进行 2H-1 的生硬映射，而是直接保存原始的 H_region
            raw_h_tensor[node_idx] = H_region
        
        # 释放内存
        torch.cuda.empty_cache() if device.type == 'cuda' else None

    # =================================================================
    # 第三步：全局-局部校准 (Z-Score + Tanh) —— 彻底解决"全正数陷阱"
    # =================================================================
    print("  正在进行相对同质性校准 (Z-Score + Tanh)...")
    
    # 计算当前图所有区域同质性的均值和标准差
    mean_h = raw_h_tensor.mean()
    std_h = raw_h_tensor.std()
    
    # Z-Score 标准化：减去均值，除以标准差 (加上 1e-8 防止除以零)
    # 低于全图平均水平的区域瞬间变成负数，高于平均水平的变成正数
    z_scores = (raw_h_tensor - mean_h) / (std_h + 1e-8)
    
    # Tanh 丝滑映射：将无界的 Z-score 完美压缩回 [-1, 1] 区间
    final_rec_tensor = torch.tanh(z_scores)
        
    return final_rec_tensor


def process_dataset(dataset_name, data_path):
    """
    处理单个数据集，计算H同质性参数并保存信息
    """
    print(f"处理数据集: {dataset_name}")
    
    # 检测并设置GPU设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print(f"  使用GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print(f"  使用CPU")
    
    # 加载数据集
    adj, feat, ano_labels, all_idx, idx_train, idx_val, idx_test, ano_labels, str_ano_labels, attr_ano_labels, normal_label_idx, abnormal_label_idx = load_mat(dataset_name, file_path=data_path)
    
    # 转换数据格式
    num_nodes = adj.shape[0]
    print(f"  总节点数: {num_nodes}")
    
    # 转换邻接矩阵为边索引
    edge_index = []
    for i, j in zip(*adj.nonzero()):
        edge_index.append([i, j])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    
    # 转换特征为张量
    if isinstance(feat, np.ndarray):
        features = torch.tensor(feat, dtype=torch.float)
    else:
        features = torch.tensor(feat.toarray(), dtype=torch.float)
    
    # 将特征和边索引移动到GPU
    features = features.to(device)
    edge_index = edge_index.to(device)
    
    # 计算相对H同质性参数 (经过 Z-score + Tanh 映射后的最终参数)
    rec_tensor = get_adaptive_regional_rec(features, edge_index, num_nodes)
    
    # 转换为numpy数组
    rec_np = rec_tensor.cpu().numpy()
    
    # 计算最终参数的统计信息
    mean_rec = np.mean(rec_np)
    std_rec = np.std(rec_np)
    min_rec = np.min(rec_np)
    max_rec = np.max(rec_np)
    
    # 这里的打印值应该呈现出漂亮的零均值分布，且必然包含负数
    print(f"  映射后权重均值: {mean_rec:.4f}")
    print(f"  映射后权重标准差: {std_rec:.4f}")
    print(f"  映射后权重最小值: {min_rec:.4f}")
    print(f"  映射后权重最大值: {max_rec:.4f}")
    
    # 创建保存目录，指向项目根目录的homophily_analysis目录
    save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'homophily_analysis')
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存所有节点的REC参数值到文件
    values_file = f"{save_dir}/{dataset_name}_homophily_values.txt"
    np.savetxt(values_file, rec_np, fmt='%.6f')
    
    print(f"  分析完成，自适应REC参数保存到: {values_file}")
    print("-" * 50)


def main():
    """
    主函数，处理所有数据集
    """
    # 数据集列表
    datasets = [
        "acm",
        "amazon",
        "blogcatalog",
        "citeseer",
        "cora",
        "elliptic",
        "facebook",
        "flickr",
        "photo",
        "pubmed",
        "reddit",
        "t_finance",
        "weibo",
        "yelpchi"
    ]
    
    # 数据集路径 - 使用相对路径指向项目根目录的datasets目录
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'datasets')
    
    # 处理每个数据集
    for dataset in datasets:
        data_path = os.path.join(data_dir, f"{dataset}.mat")
        if os.path.exists(data_path):
            process_dataset(dataset, data_path)
        else:
            print(f"数据集文件不存在: {data_path}\n")


if __name__ == "__main__":
    main()
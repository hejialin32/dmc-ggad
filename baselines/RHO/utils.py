import time
import math
import random
import numpy as np
import scipy as sp
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def init_params(module):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.01)
        if module.bias is not None:
            module.bias.data.zero_()


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = False


def get_split(num_node, label, train_rate=0.3, val_rate=0.0):

    if torch.is_tensor(label):
        all_labels = np.squeeze(label.detach().cpu().numpy())
    else:
        all_labels = np.squeeze(np.array(label))
    #num_node = index
    num_train = int(num_node * train_rate)
    num_val = int(num_node * val_rate)
    
    # 分离正常节点和异常节点
    normal_idx = [i for i in range(num_node) if all_labels[i] == 0]
    anomal_idx = [i for i in range(num_node) if all_labels[i] == 1]
    
    # 确保正常节点和异常节点都有足够的数量
    if len(normal_idx) == 0 or len(anomal_idx) == 0:
        print("警告：数据集中只有一种类型的节点！")
        # 随机划分
        all_idx = list(range(num_node))
        random.shuffle(all_idx)
        idx_train = all_idx[: num_train]
        idx_val = all_idx[num_train: num_train + num_val]
        idx_test = all_idx[num_train + num_val:]
    else:
        # 确保测试集中同时包含正常节点和异常节点
        random.shuffle(normal_idx)
        random.shuffle(anomal_idx)
        
        # 计算测试集中应该包含的正常节点和异常节点数量
        num_test = num_node - num_train - num_val
        num_test_normal = int(num_test * len(normal_idx) / num_node)
        num_test_anomal = num_test - num_test_normal
        
        # 确保至少有一个正常节点和一个异常节点
        num_test_normal = max(1, num_test_normal)
        num_test_anomal = max(1, num_test_anomal)
        
        # 划分测试集
        test_normal = normal_idx[:num_test_normal]
        test_anomal = anomal_idx[:num_test_anomal]
        idx_test = test_normal + test_anomal
        
        # 剩余的节点用于训练和验证
        remaining_normal = normal_idx[num_test_normal:]
        remaining_anomal = anomal_idx[num_test_anomal:]
        remaining_all = remaining_normal + remaining_anomal
        random.shuffle(remaining_all)
        
        idx_train = remaining_all[:num_train]
        idx_val = remaining_all[num_train:num_train+num_val]
    
    all_normal_label_idx = [i for i in idx_train if all_labels[i] == 0]
    rate = 0.5  #  change train_rate to 0.3 0.5 0.6  0.8
    normal_label_idx = all_normal_label_idx[: int(len(all_normal_label_idx) * rate)]

    
    return normal_label_idx, idx_val, idx_test

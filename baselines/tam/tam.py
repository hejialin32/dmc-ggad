# -*- coding: utf-8 -*-
import torch.nn as nn
import os
import sys
import numpy as np
import scipy.sparse as sp
import torch

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model_tam import Model
from utils.utils_tam import *
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import roc_auc_score
import argparse
from tqdm import tqdm
import time
import json
from datetime import datetime, timedelta

class TimeTracker:
    """时间追踪器 - 用于预测和显示训练时间"""
    
    def __init__(self, total_steps, description="训练"):
        self.total_steps = total_steps
        self.description = description
        self.start_time = None
        self.current_step = 0
        self.step_times = []
        self.last_step_time = None
        
    def start(self):
        """开始计时"""
        self.start_time = time.time()
        self.last_step_time = self.start_time
        print(f"\n⏱️  [{self.description}] 开始时间: {self._format_time(self.start_time)}")
        
    def update(self, current_step, additional_info=""):
        """更新进度并显示预测信息"""
        self.current_step = current_step
        now = time.time()
        
        if self.last_step_time is not None:
            step_duration = now - self.last_step_time
            self.step_times.append(step_duration)
        
        self.last_step_time = now
        
        if len(self.step_times) > 0 and current_step > 0:
            avg_time_per_step = sum(self.step_times[-min(10, len(self.step_times)):]) / min(10, len(self.step_times))
            remaining_steps = self.total_steps - current_step
            estimated_remaining = avg_time_per_step * remaining_steps
            estimated_end = now + estimated_remaining
            
            elapsed = now - self.start_time
            progress_pct = (current_step / self.total_steps) * 100
            
            print(f"\n📊 [{self.description}] 进度: {current_step}/{self.total_steps} ({progress_pct:.1f}%)")
            print(f"   ⏰ 已用时间: {self._format_duration(elapsed)}")
            print(f"   ⏳ 预计剩余: {self._format_duration(estimated_remaining)}")
            print(f"   🎯 预计结束: {self._format_time(estimated_end)}")
            
            if additional_info:
                print(f"   ℹ️  {additional_info}")
                
    def finish(self):
        """结束计时"""
        end_time = time.time()
        total_duration = end_time - self.start_time
        print(f"\n✅ [{self.description}] 完成!")
        print(f"   ⏱️  结束时间: {self._format_time(end_time)}")
        print(f"   ⏰ 总耗时: {self._format_duration(total_duration)}")
        return {
            'start_time': self._format_time(self.start_time),
            'end_time': self._format_time(end_time),
            'total_seconds': total_duration,
            'formatted_duration': self._format_duration(total_duration),
            'total_steps': self.current_step
        }
    
    def _format_time(self, timestamp):
        """格式化时间为可读字符串"""
        return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
    
    def _format_duration(self, seconds):
        """格式化持续时间为可读字符串"""
        if seconds < 60:
            return f"{seconds:.1f}秒"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}分{secs}秒"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            return f"{hours}时{minutes}分{secs}秒"

# os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, [2]))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Set argument
parser = argparse.ArgumentParser(description='Truncated Affinity Maximization for Graph Anomaly Detection')
parser.add_argument('--dataset', type=str,
                    default='photo')  # 'BlogCatalog'  'ACM'  'Amazon' 'Facebook'  'Reddit'  'YelpChi'
parser.add_argument('--lr', type=float)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--embedding_dim', type=int, default=128)
parser.add_argument('--num_epoch', type=int)
parser.add_argument('--drop_prob', type=float, default=0.0)
parser.add_argument('--subgraph_size', type=int, default=15)
parser.add_argument('--readout', type=str, default='avg')  # max min avg  weighted_sum
parser.add_argument('--margin', type=int, default=2)
parser.add_argument('--negsamp_ratio', type=int, default=2)
parser.add_argument('--cutting', type=int, default=8)  # 3 5 8 10
parser.add_argument('--N_tree', type=int, default=1)  # 3 5 8 10
parser.add_argument('--lamda', type=int, default=0)  # 0  0.5  1
parser.add_argument('--dataset_model', type=str, default='photo')  # 0  0.5  1
# 可用数据集列表
# ['facebook', 'cora', 'photo', 'acm', 'citeseer', 'dblp', 'flickr', 'tolokers', 'weibo', 'pubmed']
args = parser.parse_args()

if args.lr is None:
    args.lr = 1e-5
if args.num_epoch is None:
    args.num_epoch = 500

print('Dataset: ', args.dataset)

def run_experiment(args, seed):
    """运行单次实验
    
    Args:
        args: 命令行参数
        seed: 随机种子
    
    Returns:
        dict: 包含实验结果的字典
    """
    # 检查是否已经存在相同的实验记录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file = os.path.join(script_dir, f'{os.path.splitext(os.path.basename(__file__))[0]}_result.csv')
    
    # 检查CSV文件是否存在
    if os.path.exists(csv_file):
        import csv
        with open(csv_file, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # 检查是否存在相同的实验记录
                if (row['数据集'] == args.dataset and 
                    row['种子'] == str(seed) and 
                    float(row['学习率']) == args.lr and 
                    float(row['权重衰减']) == args.weight_decay and 
                    int(row['嵌入维度']) == args.embedding_dim and 
                    float(row['丢弃概率']) == args.drop_prob and 
                    int(row['子图大小']) == args.subgraph_size and 
                    row['读出方式'] == args.readout and 
                    int(row['边界']) == args.margin and 
                    int(row['负采样比例']) == args.negsamp_ratio and 
                    int(row['截断']) == args.cutting and 
                    int(row['树数量']) == args.N_tree and 
                    int(row['lambda']) == args.lamda):
                    print(f"\n{'='*60}")
                    print(f'⚠️  跳过实验 - 已存在相同记录')
                    print(f'   数据集: {args.dataset}')
                    print(f'   种子: {seed}')
                    print(f"{'='*60}")
                    return {
                        'seed': seed,
                        'best_auc': float(row['最佳AUC']),
                        'best_ap': float(row['最佳AP']),
                        'best_epoch': int(row['最佳AUC轮次'])
                    }
    
    print(f"\n{'='*60}")
    print(f'🚀 开始实验 - Seed: {seed}')
    print(f"{'='*60}")
    
    # 初始化时间追踪器
    total_training_steps = args.cutting * args.num_epoch
    time_tracker = TimeTracker(total_steps=total_training_steps, description=f"实验 {args.dataset}-Seed{seed}")
    time_tracker.start()
    
    # Set random seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['OMP_NUM_THREADS'] = '1'
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Load and preprocess data
    adj, features, ano_label, str_ano_label, attr_ano_label, normal_label_idx, idx_test = load_mat(args.dataset)

    if args.dataset in [ 'YelpChi', 'Amazon-all', 'YelpChi-all', 'elliptic_no_isolate']:
        features, _ = preprocess_features(features)
        raw_features = features
    else:
        raw_features = features.todense()
        features = raw_features

    nb_nodes = features.shape[0]
    ft_size = features.shape[1]
    raw_adj = adj
    print(raw_adj.sum())
    raw_adj = (raw_adj + sp.eye(adj.shape[0])).todense()

    adj = (adj + sp.eye(adj.shape[0])).todense()
    raw_features = torch.FloatTensor(raw_features[np.newaxis])
    features = torch.FloatTensor(features[np.newaxis])
    adj = torch.FloatTensor(adj[np.newaxis])
    raw_adj = torch.FloatTensor(raw_adj[np.newaxis])

    # Initialize model and optimiser
    optimiser_list = []
    model_list = []
    for i in range(args.cutting * args.N_tree):
        model = Model(ft_size, args.embedding_dim, 'prelu', args.negsamp_ratio, args.readout)
        optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimiser_list.append(optimiser)
        model_list.append(model)

    criterion = nn.CrossEntropyLoss()

    # Variables to track best results
    best_auc = 0.0
    best_ap = 0.0
    best_epoch = 0
    patience = 120  # 早停止的耐心值（连续多少轮次没有提升就停止）
    best_score = None

    start = time.time()
    # Train model
    with tqdm(total=args.num_epoch) as pbar:
        pbar.set_description('Training')

        score_list = []
        new_adj_list = []
        for n_t in range(args.N_tree):
            new_adj_list.append(raw_adj)
        all_cut_adj = torch.cat(new_adj_list)
        origin_degree = torch.sum(torch.squeeze(raw_adj), 0)
        print('<<<<<<Start to calculate distance<<<<<')
        # Create distance_save directory if it doesn't exist
        distance_save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'utils', 'distance_save')
        if not os.path.exists(distance_save_dir):
            os.makedirs(distance_save_dir)
        dis_path = os.path.join(distance_save_dir, "dis_array_{}.npy".format(args.dataset))

        dis_array = None
        if os.path.exists(dis_path):
            loaded_dis_array = torch.from_numpy(np.load(dis_path))
            # Check if the loaded dis_array matches the current dataset dimensions
            if loaded_dis_array.shape[0] == nb_nodes and loaded_dis_array.shape[1] == nb_nodes:
                dis_array = loaded_dis_array
                print(f'Loaded cached dis_array from {dis_path}')
            else:
                print(f'Cached dis_array dimension mismatch: {loaded_dis_array.shape} vs ({nb_nodes}, {nb_nodes})')
                print(f'Recalculating dis_array for {args.dataset}...')
        
        if dis_array is None:
            dis_array = calc_distance(raw_adj[0, :, :], raw_features[0, :, :])
            np.save(dis_path, dis_array)
            print(f'Saved dis_array to {dis_path}')

        index = 0
        message_mean_list = []
        for n_cut in range(args.cutting):
            print('n_cut.{}'.format(n_cut))
            
            # 更新时间追踪器 - 每个cutting阶段开始时
            current_step = n_cut * args.num_epoch
            time_tracker.update(current_step, additional_info=f"Cutting阶段: {n_cut + 1}/{args.cutting}")
            
            feat_list = []
            message_list = []
            for n_t in range(args.N_tree):
                cut_adj = graph_nsgt(dis_array, all_cut_adj[n_t, :, :])
                cut_adj = cut_adj.unsqueeze(0)
                optimiser_list[index].zero_grad()
                model_list[index].train()
                print("<<<< cutting num .{}<<<<<<".format(n_cut))
                adj_norm = normalize_adj_tensor(cut_adj)

                for epoch in range(args.num_epoch):
                    all_idx = list(range(nb_nodes))

                    node_emb, feat1, feat2 = model_list[index].forward(features, adj_norm)
                    # maximize the message flow
                    loss, message_sum1 = max_message(node_emb[0, :, :], raw_adj[0, :, :], normal_label_idx)

                    message_sum = inference(node_emb[0, :, :], raw_adj[0, :, :])
                    loss.backward()
                    optimiser_list[index].step()

                    loss = loss.detach().cpu().numpy()

                    if epoch % 50 == 0:
                        print("mean_loss is {}".format(loss))
                message_list.append(torch.unsqueeze(message_sum, 0))
                all_cut_adj[n_t, :, :] = torch.squeeze(cut_adj)
                index += 1

            for mes in message_list:
                mes = np.array(torch.squeeze(mes).cpu().detach())
                mes = 1 - normalize_score(mes)
                auc = roc_auc_score(ano_label, mes)
                print('{} AUC:{:.4f}'.format(args.dataset, auc))

            message_list = torch.mean(torch.cat(message_list), 0)
            message_mean_list.append(torch.unsqueeze(message_list, 0))

            message = np.array(message_list.cpu().detach())
            adj_array = np.array(raw_adj[0, :, :].cpu().detach())

            message = 1 - normalize_score(message)

            score = message
            auc = roc_auc_score(ano_label[idx_test], score[idx_test])
            AP = average_precision_score(ano_label[idx_test], score[idx_test], average='macro', pos_label=1, sample_weight=None)
            print('AP:', AP)
            print('{} AUC:{:.4f}'.format(args.dataset, auc))
            
            # Update best AUC and AP
            if auc > best_auc:
                best_auc = auc
                best_ap = AP
                best_epoch = n_cut
                best_score = score
                print('New best AUC: {:.4f}, AP: {:.4f} at cutting {}'.format(best_auc, best_ap, best_epoch))

            message_mean_cut = torch.mean(torch.cat(message_mean_list), 0)
            message_mean = np.array(message_mean_cut.cpu().detach())
            message_mean = 1 - normalize_score(message_mean)
            score = message_mean
            auc = roc_auc_score(ano_label, score)
            AP = average_precision_score(ano_label, score, average='macro', pos_label=1, sample_weight=None)
            print('AP:', AP)
            print('{} AUC:{:.4f}'.format(args.dataset, auc))
            
            # Update best AUC and AP for aggregated score
            if auc > best_auc:
                best_auc = auc
                best_ap = AP
                best_epoch = n_cut
                best_score = score
                print('New best AUC (aggregated): {:.4f}, AP: {:.4f} at cutting {}'.format(best_auc, best_ap, best_epoch))
            
            # 早停止检查：如果连续200个cutting阶段没有超越最佳AUC，则停止训练
            if n_cut - best_epoch >= patience:
                print(f"\n[早停止] 连续 {patience} 个cutting阶段没有超越最佳AUC {best_auc:.4f}，训练提前结束于cutting {n_cut}")
                break

        # 获取时间统计信息
        time_stats = time_tracker.finish()
        
        end = time.time()
        print(end - start)

    # Print best results after training
    print('\n=============================================')
    print('Best AUC: {:.4f}, AP: {:.4f} at cutting {}'.format(best_auc, best_ap, best_epoch))
    print('=============================================')

    # 生成运行命令
    run_command = f'python {os.path.relpath(os.path.abspath(__file__), os.path.dirname(os.path.dirname(os.path.abspath(__file__))))} --dataset {args.dataset}'
    run_command += f' --seed {seed}'
    if args.lr is not None:
        run_command += f' --lr {args.lr}'
    run_command += f' --weight_decay {args.weight_decay}'
    run_command += f' --num_epoch {args.num_epoch}'
    run_command += f' --embedding_dim {args.embedding_dim}'
    run_command += f' --drop_prob {args.drop_prob}'
    run_command += f' --subgraph_size {args.subgraph_size}'
    run_command += f' --readout {args.readout}'
    run_command += f' --margin {args.margin}'
    run_command += f' --negsamp_ratio {args.negsamp_ratio}'
    run_command += f' --cutting {args.cutting}'
    run_command += f' --N_tree {args.N_tree}'
    run_command += f' --lamda {args.lamda}'
    run_command += f' --dataset_model {args.dataset_model}'

    # 算法名称 - 使用当前脚本文件的名称（去掉.py后缀）
    algorithm_name = os.path.splitext(os.path.basename(__file__))[0]

    # 生成时间戳
    time_str = time.strftime('%m%d%H%M%S', time.localtime())

    # 格式化AUC值为字符串
    auc_str = f'{best_auc:.3f}'

    # 保存结果到同目录的唯一CSV文件
    import csv
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file = os.path.join(script_dir, f'{algorithm_name}_result.csv')
    fieldnames = ['数据集', '算法', '种子', '最佳AUC', '最佳AP', '最佳AUC轮次', '总轮次', '学习率', '权重衰减', '嵌入维度', '丢弃概率', '子图大小', '读出方式', '边界', '负采样比例', '截断', '树数量', 'lambda', '数据集模型', '开始时间', '结束时间', '总耗时(秒)', '格式化耗时', '时间戳']
    
    # 检查文件是否存在
    file_exists = os.path.exists(csv_file)
    
    with open(csv_file, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # 如果文件不存在，写入表头
        if not file_exists:
            writer.writeheader()
        
        # 写入数据
        csv_row = {
            '数据集': args.dataset,
            '算法': algorithm_name,
            '种子': seed,
            '最佳AUC': float(best_auc),
            '最佳AP': float(best_ap),
            '最佳AUC轮次': best_epoch,
            '总轮次': args.num_epoch,
            '学习率': args.lr,
            '权重衰减': args.weight_decay,
            '嵌入维度': args.embedding_dim,
            '丢弃概率': args.drop_prob,
            '子图大小': args.subgraph_size,
            '读出方式': args.readout,
            '边界': args.margin,
            '负采样比例': args.negsamp_ratio,
            '截断': args.cutting,
            '树数量': args.N_tree,
            'lambda': args.lamda,
            '数据集模型': args.dataset_model,
            '开始时间': time_stats.get('start_time', ''),
            '结束时间': time_stats.get('end_time', ''),
            '总耗时(秒)': f"{time_stats.get('total_seconds', 0):.2f}",
            '格式化耗时': time_stats.get('formatted_duration', ''),
            '时间戳': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        }
        writer.writerow(csv_row)
    
    print(f'[保存] 结果已追加到CSV文件: {csv_file}')
    
    # 返回实验结果
    return {
        'seed': seed,
        'best_auc': float(best_auc),
        'best_ap': float(best_ap),
        'best_epoch': best_epoch,
        **time_stats  # 添加时间统计信息
    }

if __name__ == '__main__':
    seed = 123
    print(f"\n{'#'*70}")
    print(f'# 运行种子: {seed}')
    print(f"{'#'*70}")
    try:
        result = run_experiment(args, seed)
        print(f"Seed {seed} 完成 - AUC: {result['best_auc']:.4f}, AP: {result['best_ap']:.4f}")
    except Exception as e:
        print(f"Seed {seed} 失败: {str(e)}")
        import traceback
        traceback.print_exc()

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import os
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_AEGIS import Model
from utils.utils import *

from sklearn.metrics import roc_auc_score
import random
from sklearn.metrics import precision_recall_curve, average_precision_score
import argparse
from tqdm import tqdm
import json

# os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, [2]))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# Set argument
parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset', type=str,
                    default='reddit')  #  'BlogCatalog'  'Flickr'  'ACM'  'cora'  'citeseer'  'pubmed'
parser.add_argument('--lr', type=float)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--embedding_dim', type=int, default=300)
parser.add_argument('--num_epoch', type=int)
parser.add_argument('--recon_num_epoch', type=int, default=10)
parser.add_argument('--drop_prob', type=float, default=0.0)
parser.add_argument('--batch_size', type=int, default=300)
parser.add_argument('--subgraph_size', type=int, default=4)
parser.add_argument('--readout', type=str, default='avg')  # max min avg  weighted_sum
parser.add_argument('--auc_test_rounds', type=int, default=256)
parser.add_argument('--negsamp_ratio', type=int, default=1)
parser.add_argument('--train_rate', type=float, default=0.3)
parser.add_argument('--val_rate', type=float, default=0.0)

args = parser.parse_args()

# 可用数据集列表
# ['facebook', 'cora', 'photo', 'acm', 'citeseer', 'dblp', 'flickr', 'tolokers', 'weibo', 'pubmed']

if args.lr is None:
    if args.dataset in ['Amazon']:
        args.lr = 1e-3
    elif args.dataset in ['t_finance']:
        args.lr = 5e-4
    elif args.dataset in ['reddit']:
        args.lr = 1e-3
    else:
        args.lr = 1e-3  # Default learning rate

if args.num_epoch is None:
    if args.dataset in ['reddit']:
        args.num_epoch = 500
    elif args.dataset in ['t_finance']:
        args.num_epoch = 1500
    elif args.dataset in ['Amazon']:
        args.num_epoch = 800
    else:
        args.num_epoch = 300  # Default number of epochs

batch_size = args.batch_size
subgraph_size = args.subgraph_size

print('Dataset: ', args.dataset)

# 定义训练函数，支持单种子训练
def train_with_seed(seed, args):
    """使用指定种子训练模型并返回结果"""
    # 检查是否已经存在相同的实验记录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file = os.path.join(script_dir, f'{os.path.splitext(os.path.basename(__file__))[0]}_result.csv')
    
    # 检查CSV文件是否存在
    if os.path.exists(csv_file):
        import csv
        with open(csv_file, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                try:
                    row_dataset = row.get('数据集', '').strip()
                    row_seed = row.get('种子', '').strip()
                    row_auc = row.get('最佳AUC', '').strip()

                    if not all([row_dataset, row_seed, row_auc]):
                        continue

                    if row_dataset == args.dataset and row_seed == str(seed) and float(row_auc) > 0:
                        print(f"\n{'='*60}")
                        print(f'[跳过] 已存在相同记录')
                        print(f'   数据集: {args.dataset}')
                        print(f'   种子: {seed}')
                        print(f"{'='*60}")
                        return {
                            'seed': seed,
                            'best_auc': float(row['最佳AUC']),
                            'best_ap': float(row['最佳AP']),
                            'best_epoch': int(row['最佳AUC轮次'])
                        }
                except (ValueError, KeyError):
                    continue
    
    print(f"\n{'='*60}")
    print(f"开始种子 {seed} 的实验")
    print(f"{'='*60}")
    
    # Set random seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    # Load and preprocess data
    adj, features, labels, all_idx, idx_train, idx_val, \
    idx_test, ano_label, str_ano_label, attr_ano_label, normal_label_idx, abnormal_label_idx = load_mat(
        args.dataset,
        train_rate=args.train_rate,
        val_rate=args.val_rate
    )
    if args.dataset in [ 'tf_finace',  'elliptic']:
        features, _ = preprocess_features(features)
    else:
        features = features.todense()

    nb_nodes = features.shape[0]
    ft_size = features.shape[1]
    use_sparse_adj = args.dataset == 'elliptic'
    raw_adj = adj
    adj = normalize_adj(adj)


    features = torch.FloatTensor(features[np.newaxis])
    if use_sparse_adj:
        raw_adj = sparse_mx_to_torch_sparse_tensor(raw_adj + sp.eye(raw_adj.shape[0])).coalesce()
        adj = sparse_mx_to_torch_sparse_tensor(adj + sp.eye(adj.shape[0])).coalesce()
    else:
        raw_adj = (raw_adj + sp.eye(raw_adj.shape[0])).todense()
        adj = (adj + sp.eye(adj.shape[0])).todense()
        adj = torch.FloatTensor(adj[np.newaxis])
        raw_adj = torch.FloatTensor(raw_adj[np.newaxis])
    labels = torch.FloatTensor(labels[np.newaxis])
    # idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)

    # Initialize model and optimiser
    model = Model(ft_size, args.embedding_dim, 'prelu', args.negsamp_ratio, args.readout)
    ae_lr = 1e-4 if args.dataset.lower() == 'dblp' else 1e-3
    optimiser_ae = torch.optim.Adam(model.parameters(), lr=ae_lr, weight_decay=args.weight_decay)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimiser_gen = torch.optim.Adam(model.generator.parameters(),
                                     lr=args.lr)
    if torch.cuda.is_available():
        print('Using CUDA')
        model.cuda()
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()

        # idx_train = idx_train.cuda()
        # idx_val = idx_val.cuda()
        # idx_test = idx_test.cuda()

    cnt_wait = 0
    best = 1e9
    best_t = 0
    best_auc = 0.0
    best_ap = 0.0
    best_epoch = 0
    patience = 120  # 早停止的耐心值（连续多少轮次没有提升就停止）
    batch_num = nb_nodes // batch_size + 1
    import time

    # Train model
    with tqdm(total=args.num_epoch) as pbar:
        pbar.set_description(f'Training (Seed: {seed})')

        for epoch in range(args.recon_num_epoch):
            optimiser_ae.zero_grad()
            loss_dis, loss_g, loss_ae, score_test, emb_all = model(features, adj, normal_label_idx, idx_test, sparse=use_sparse_adj)
            # loss_dis, loss_g, loss_ae, score_test, emb_all = model(features, adj, all_idx, idx_test)
            if not torch.isfinite(loss_ae):
                print(f"[警告] ae_loss 非有限，停止预训练: epoch={epoch}, loss={loss_ae.item()}")
                break
            loss_ae.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimiser_ae.step()
            print("Epoch:", '%04d' % (epoch), "ae_loss=", "{:.5f}".format(loss_ae.item()))
        total_time = 0
        for epoch in range(args.num_epoch):
            start_time = time.time()
            model.train()
            optimiser.zero_grad()
            optimiser_gen.zero_grad()
            # Train model
            loss_dis, loss_g, loss_ae, score_test, emb_all = model(features, adj, normal_label_idx, idx_test, sparse=use_sparse_adj)
            loss_g.backward(retain_graph=True)
            loss_dis.backward(retain_graph=True)
            # loss = loss_dis + loss_g
            optimiser.step()
            optimiser_gen.step()
            score_test = np.array(score_test.detach().cpu())

            if not use_sparse_adj:
                emb_inf = torch.norm(emb_all, dim=-1, keepdim=True)
                emb_inf = torch.pow(emb_inf, -1)
                emb_inf[torch.isinf(emb_inf)] = 0.
                emb_norm = emb_all * emb_inf

                sim_matrix = torch.mm(emb_norm, emb_norm.T)
                raw_adj_cuda = torch.squeeze(raw_adj).cuda() if torch.cuda.is_available() else torch.squeeze(raw_adj)
                similar_matrix1 = sim_matrix[:int(raw_adj_cuda.shape[0]), :int(raw_adj_cuda.shape[1])] * raw_adj_cuda
                similar_matrix2 = sim_matrix[int(raw_adj_cuda.shape[0]):, int(raw_adj_cuda.shape[1]):] * raw_adj_cuda

                r_inv = torch.pow(torch.sum(raw_adj_cuda, 0), -1)
                r_inv[torch.isinf(r_inv)] = 0.
                affinity1 = torch.sum(similar_matrix1, 0) * r_inv
                affinity2 = torch.sum(similar_matrix2, 0) * r_inv

            if epoch % 5 == 0:
                print("Epoch:", '%04d' % (epoch), "train_loss=", "{:.5f}".format(loss_dis.item()))
                model.eval()
                auc = roc_auc_score(ano_label[idx_test], score_test)
                print('Testing {} AUC:{:.4f}'.format(args.dataset, auc))
                AP = average_precision_score(ano_label[idx_test], score_test, average='macro', pos_label=1,
                                             sample_weight=None)
                print('Testing AP:', AP)
                print('Total time is', total_time)
                
                # Update best AUC and AP
                if auc > best_auc:
                    best_auc = auc
                    best_ap = AP
                    best_epoch = epoch
                    print('New best AUC: {:.4f}, AP: {:.4f} at epoch {}'.format(best_auc, best_ap, best_epoch))
                
                # 早停止检查：如果连续200轮次没有超越最佳AUC，则停止训练
                if epoch - best_epoch >= patience:
                    print(f"\n[早停止] 连续 {patience} 轮次没有超越最佳AUC {best_auc:.4f}，训练提前结束于epoch {epoch}")
                    break

            end_time = time.time()
            total_time += end_time - start_time
            pbar.update(1)

    # Print best results after training
    print('\n=============================================')
    print('Best AUC: {:.4f}, AP: {:.4f} at epoch {}'.format(best_auc, best_ap, best_epoch))
    print('=============================================')

    # 生成运行命令
    run_command = f'python {os.path.basename(__file__)} --dataset {args.dataset}'
    run_command += f' --seed {seed}'
    if args.lr is not None:
        run_command += f' --lr {args.lr}'
    run_command += f' --num_epoch {args.num_epoch}'
    run_command += f' --recon_num_epoch {args.recon_num_epoch}'
    run_command += f' --embedding_dim {args.embedding_dim}'
    run_command += f' --drop_prob {args.drop_prob}'
    run_command += f' --batch_size {args.batch_size}'
    run_command += f' --subgraph_size {args.subgraph_size}'
    run_command += f' --readout {args.readout}'
    run_command += f' --auc_test_rounds {args.auc_test_rounds}'
    run_command += f' --negsamp_ratio {args.negsamp_ratio}'
    run_command += f' --train_rate {args.train_rate}'
    run_command += f' --val_rate {args.val_rate}'

    # 算法名称 - 使用当前脚本文件的名称（去掉.py后缀）
    algorithm_name = os.path.splitext(os.path.basename(__file__))[0]

    # 生成时间戳
    time_str = time.strftime('%m%d%H%M%S', time.localtime())

    # 格式化AUC值为字符串
    auc_str = f'{best_auc:.3f}'

    # 计算平均每轮用时
    avg_epoch_time = round(total_time / (epoch + 1), 5) if epoch > 0 else 0
    
    # 保存结果到同目录的唯一CSV文件
    import csv
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file = os.path.join(script_dir, f'{algorithm_name}_result.csv')
    fieldnames = ['数据集', '算法', '种子', '最佳AUC', '最佳AP', '最佳AUC轮次', '总轮次', '重构轮次', '学习率', '嵌入维度', '丢弃概率', '批量大小', '子图大小', '读出方式', 'AUC测试轮次', '负采样比例', '训练比例', '验证比例', '训练索引范围', '每轮用时', '时间戳']
    
    # 检查文件是否存在
    file_exists = os.path.exists(csv_file)

    if file_exists:
        with open(csv_file, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            existing_rows = list(reader)
            existing_fieldnames = reader.fieldnames or []
        if existing_fieldnames != fieldnames:
            with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for row in existing_rows:
                    writer.writerow({field: row.get(field, '') for field in fieldnames})
    
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
            '重构轮次': args.recon_num_epoch,
            '学习率': args.lr,
            '嵌入维度': args.embedding_dim,
            '丢弃概率': args.drop_prob,
            '批量大小': args.batch_size,
            '子图大小': args.subgraph_size,
            '读出方式': args.readout,
            'AUC测试轮次': args.auc_test_rounds,
            '负采样比例': args.negsamp_ratio,
            '训练比例': args.train_rate,
            '验证比例': args.val_rate,
            '训练索引范围': 'normal_label_idx',
            '每轮用时': avg_epoch_time,
            '时间戳': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        }
        writer.writerow(csv_row)
    
    print(f'[保存] 结果已追加到CSV文件: {csv_file}')
    
    # 返回结果
    return {
        'seed': seed,
        'best_auc': float(best_auc),
        'best_ap': float(best_ap),
        'best_epoch': best_epoch
    }

# 主函数
if __name__ == "__main__":
    seed = 123
    print(f"\n{'#'*70}")
    print(f'# 运行种子: {seed}')
    print(f"{'#'*70}")
    try:
        result = train_with_seed(seed, args)
        print(f"Seed {seed} 完成 - AUC: {result['best_auc']:.4f}, AP: {result['best_ap']:.4f}")
    except Exception as e:
        print(f"Seed {seed} 失败: {str(e)}")
        import traceback
        traceback.print_exc()

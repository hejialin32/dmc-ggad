import numpy as np
import scipy.sparse as sp
import torch
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_AnomalyDAE import Model
from utils.utils import *

from sklearn.metrics import roc_auc_score
import random
from sklearn.metrics import  average_precision_score
import argparse
from tqdm import tqdm
import json
import time

parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset', type=str,
                    default='t_finance')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--embedding_dim', type=int, default=300)
parser.add_argument('--num_epoch', type=int, default=300)
parser.add_argument('--drop_prob', type=float, default=0.0)
parser.add_argument('--batch_size', type=int, default=300)
parser.add_argument('--subgraph_size', type=int, default=4)
parser.add_argument('--readout', type=str, default='avg')
parser.add_argument('--auc_test_rounds', type=int, default=256)
parser.add_argument('--negsamp_ratio', type=int, default=1)

# 可用数据集列表
# ['facebook', 'cora', 'photo', 'acm', 'citeseer', 'dblp', 'flickr', 'tolokers', 'weibo', 'pubmed']

args = parser.parse_args()

if args.lr is None:
    if args.dataset in ['Amazon']:
        args.lr = 1e-3
    elif args.dataset in ['t_finance']:
        args.lr = 5e-4
    elif args.dataset in ['reddit']:
        args.lr = 1e-3
    elif args.dataset in ['photo']:
        args.lr = 3e-3
    elif args.dataset in ['elliptic']:
        args.lr = 3e-3

if args.num_epoch is None:

    if args.dataset in ['reddit']:
        args.num_epoch = 500
    elif args.dataset in ['t_finance']:
        args.num_epoch = 1500
    elif args.dataset in ['Amazon']:
        args.num_epoch = 800
    elif args.dataset in ['photo']:
        args.num_epoch = 500
    elif args.dataset in ['elliptic']:
        args.num_epoch = 500

batch_size = args.batch_size
subgraph_size = args.subgraph_size

print('Dataset: ', args.dataset)

def run_experiment(args, seed):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file = os.path.join(script_dir, f'{os.path.splitext(os.path.basename(__file__))[0]}_result.csv')
    
    if os.path.exists(csv_file):
        import csv
        with open(csv_file, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if (float(row['最佳AUC']) > 0 and
                    row['数据集'] == args.dataset and 
                    row['种子'] == str(seed) and 
                    float(row['学习率']) == args.lr and 
                    float(row['权重衰减']) == args.weight_decay and 
                    int(row['嵌入维度']) == args.embedding_dim and 
                    float(row['丢弃概率']) == args.drop_prob and 
                    int(row['批量大小']) == args.batch_size and 
                    int(row['子图大小']) == args.subgraph_size and 
                    row['读出方式'] == args.readout and 
                    int(row['AUC测试轮次']) == args.auc_test_rounds and 
                    int(row['负采样比例']) == args.negsamp_ratio):
                    print(f"\n{'='*60}")
                    print(f'⚠️  跳过实验 - 已存在相同有效记录 (AUC={row["最佳AUC"]})')
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
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    adj, features, labels, all_idx, idx_train, idx_val, \
    idx_test, ano_label, str_ano_label, attr_ano_label, normal_label_idx, abnormal_label_idx = load_mat(args.dataset)

    if args.dataset in [ 'tf_finace',  'elliptic']:
        features, _ = preprocess_features(features)
    else:
        features = features.todense()

    nb_nodes = features.shape[0]
    ft_size = features.shape[1]
    raw_adj = adj
    adj = normalize_adj(adj)
    adj = adj + sp.eye(adj.shape[0])
    raw_adj = raw_adj + sp.eye(raw_adj.shape[0])

    def sparse_to_torch(sp_mat):
        coo = sp_mat.tocoo()
        i = torch.LongTensor(np.vstack((coo.row, coo.col)))
        v = torch.FloatTensor(coo.data)
        return torch.sparse_coo_tensor(i, v, coo.shape)

    adj_sparse = sparse_to_torch(adj)
    raw_adj_sparse = sparse_to_torch(raw_adj)
    features = torch.FloatTensor(features[np.newaxis])
    labels = torch.FloatTensor(labels[np.newaxis])

    model = Model(ft_size, args.embedding_dim, 'prelu', args.negsamp_ratio, args.readout)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if torch.cuda.is_available():
        print('Using CUDA')
        model.cuda()
        features = features.cuda()
        adj = adj_sparse.to_dense().unsqueeze(0).cuda()
        labels = labels.cuda()

    cnt_wait = 0
    best = 1e9
    best_t = 0
    best_auc = 0.0
    best_ap = 0.0
    best_epoch = 0
    patience = 120
    batch_num = nb_nodes // batch_size + 1

    scaler = torch.cuda.amp.GradScaler()

    with tqdm(total=args.num_epoch) as pbar:
        total_time = 0
        pbar.set_description('Training')
        for epoch in range(args.num_epoch):
            start_time = time.time()
            model.train()
            optimiser.zero_grad()

            with torch.cuda.amp.autocast():
                loss, score = model(features, adj, normal_label_idx, idx_test)
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()

            if epoch % 2 == 0:
                print("Epoch:", '%04d' % (epoch), "train_loss=", "{:.5f}".format(loss.item()))

            if epoch % 5 == 0:
                 model.eval()
                 score = np.array(score.detach().cpu())
                 auc = roc_auc_score(ano_label[idx_test], score)
                 print('Testing {} AUC:{:.4f}'.format(args.dataset, auc))
                 AP = average_precision_score(ano_label[idx_test], score, average='macro', pos_label=1, sample_weight=None)
                 print('Testing AP:', AP)
                 
                 if auc > best_auc:
                     best_auc = auc
                     best_ap = AP
                     best_epoch = epoch
                     print('New best AUC: {:.4f}, AP: {:.4f} at epoch {}'.format(best_auc, best_ap, best_epoch))
                 
                 if epoch - best_epoch >= patience:
                     print(f"\n[早停止] 连续 {patience} 轮次没有超越最佳AUC {best_auc:.4f}，训练提前结束于epoch {epoch}")
                     break
                 print('Total time is', total_time)

            end_time = time.time()
            total_time += end_time - start_time

    print('\n=============================================')
    print('Best AUC: {:.4f}, AP: {:.4f} at epoch {}'.format(best_auc, best_ap, best_epoch))
    print('=============================================')

    algorithm_name = os.path.splitext(os.path.basename(__file__))[0]

    import csv
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file = os.path.join(script_dir, f'{algorithm_name}_result.csv')
    fieldnames = ['数据集', '算法', '种子', '最佳AUC', '最佳AP', '最佳AUC轮次', '总轮次', '学习率', '权重衰减', '嵌入维度', '丢弃概率', '批量大小', '子图大小', '读出方式', 'AUC测试轮次', '负采样比例', '时间戳']
    
    file_exists = os.path.exists(csv_file)
    
    with open(csv_file, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
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
            '批量大小': args.batch_size,
            '子图大小': args.subgraph_size,
            '读出方式': args.readout,
            'AUC测试轮次': args.auc_test_rounds,
            '负采样比例': args.negsamp_ratio,
            '时间戳': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        }
        writer.writerow(csv_row)
    
    print(f'[保存] 结果已追加到CSV文件: {csv_file}')
    
    return {
        'seed': seed,
        'best_auc': float(best_auc),
        'best_ap': float(best_ap),
        'best_epoch': best_epoch
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

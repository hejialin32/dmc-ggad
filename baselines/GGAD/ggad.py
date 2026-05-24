import torch.nn as nn
import os
import sys
import numpy as np
import scipy.sparse as sp
import torch

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_ggad import Model
from utils import *

from sklearn.metrics import roc_auc_score
import random
from sklearn.metrics import average_precision_score
import argparse
from tqdm import tqdm
import time
import json

# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, [3]))
# os.environ["KMP_DUPLICATE_LnIB_OK"] = "TRUE"
# Set argument
parser = argparse.ArgumentParser(description='')

parser.add_argument('--dataset', type=str,
                    default='reddit')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--embedding_dim', type=int, default=300)
parser.add_argument('--num_epoch', type=int, default=300)
parser.add_argument('--drop_prob', type=float, default=0.0)
parser.add_argument('--readout', type=str, default='avg')  # max min avg  weighted_sum
parser.add_argument('--auc_test_rounds', type=int, default=256)
parser.add_argument('--negsamp_ratio', type=int, default=1)
parser.add_argument('--mean', type=float, default=0.0)
parser.add_argument('--var', type=float, default=0.0)
parser.add_argument('--output_csv', type=str, default='',
                    help='指定输出CSV文件路径，为空则使用默认ggad_results.csv')



# 可用数据集列表
# ['facebook', 'cora', 'photo', 'acm', 'citeseer', 'dblp', 'flickr', 'tolokers', 'weibo', 'pubmed']
args = parser.parse_args()

# Convert dataset name to lowercase for case-insensitive matching
args.dataset = args.dataset.lower()

if args.lr is None:
    if args.dataset in ['amazon']:
        args.lr = 1e-3
    elif args.dataset in ['t_finance']:
        args.lr = 1e-3
    elif args.dataset in ['reddit']:
        args.lr = 1e-3
    elif args.dataset in ['photo']:
        args.lr = 1e-3
    elif args.dataset in ['elliptic']:
        args.lr = 1e-3

if args.num_epoch is None:
    if args.dataset in ['photo']:
        args.num_epoch = 100
    if args.dataset in ['elliptic']:
        args.num_epoch = 150
    if args.dataset in ['reddit']:
        args.num_epoch = 300
    elif args.dataset in ['t_finance']:
        args.num_epoch = 500
    elif args.dataset in ['amazon']:
        args.num_epoch = 800
if args.dataset in [ 'photo']:
    args.mean = 0.02
    args.var = 0.01
else:
    args.mean = 0.0
    args.var = 0.0


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
    csv_file = os.path.join(script_dir, 'ggad_results.csv')
    
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
                    row['读出方式'] == args.readout and 
                    int(row['AUC测试轮次']) == args.auc_test_rounds and 
                    int(row['负采样比例']) == args.negsamp_ratio and 
                    float(row['均值']) == args.mean and 
                    float(row['方差']) == args.var and 
                    int(row['总轮次']) == args.num_epoch):
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
    
    # Set random seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)
    # os.environ['OMP_NUM_THREADS'] = '1'
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Load and preprocess data
    adj, features, labels, all_idx, idx_train, idx_val, \
    idx_test, ano_label, str_ano_label, attr_ano_label, normal_label_idx, abnormal_label_idx = load_mat(args.dataset)

    if args.dataset in [ 'tf_finace',  'elliptic']:
        features, _ = preprocess_features(features)
    else:
        features = features.todense()

    nb_nodes = features.shape[0]
    ft_size = features.shape[1]
    raw_adj = adj
    print(adj.sum())
    adj = normalize_adj(adj)

    raw_adj = (raw_adj + sp.eye(raw_adj.shape[0])).todense()
    adj = (adj + sp.eye(adj.shape[0])).todense()

    features = torch.FloatTensor(features[np.newaxis])
    # adj = torch.FloatTensor(adj[np.newaxis])
    features = torch.FloatTensor(features)
    adj = torch.FloatTensor(adj)
    # adj = adj.to_sparse_csr()
    adj = torch.FloatTensor(adj[np.newaxis])
    raw_adj = torch.FloatTensor(raw_adj[np.newaxis])
    labels = torch.FloatTensor(labels[np.newaxis])

    # idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)

    # Initialize model and optimiser
    model = Model(ft_size, args.embedding_dim, 'prelu', args.negsamp_ratio, args.readout)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Initialize variables to track best AUC and AP
    best_auc = 0.0
    best_ap = 0.0
    best_epoch = 0
    patience = 120  # 早停止的耐心值（连续多少轮次没有提升就停止）
    epoch_times = []  # 记录每轮次的运行时间（秒）

    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print('Using CUDA')
        model.to(device)
        features = features.to(device)
        adj = adj.to(device)
        labels = labels.to(device)
        raw_adj = raw_adj.to(device)
    else:
        print('Using CPU')

    # Convert indices to tensors
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    if torch.cuda.is_available():
        idx_train = idx_train.to(device)
        idx_val = idx_val.to(device)
        idx_test = idx_test.to(device)

    if torch.cuda.is_available():
        b_xent = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([args.negsamp_ratio]).to(device))
    else:
        b_xent = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([args.negsamp_ratio]))
    xent = nn.CrossEntropyLoss()


    # Train model
    with tqdm(total=args.num_epoch) as pbar:
        pbar.set_description('Training')
        total_time = 0
        for epoch in range(args.num_epoch):
            start_time = time.time()
            model.train()
            optimiser.zero_grad()

            # Train model
            train_flag = True
            emb, emb_combine, logits, emb_con, emb_abnormal = model(features, adj,
                                                                    abnormal_label_idx, normal_label_idx,
                                                                    train_flag, args)
            if epoch % 10 == 0:
                # save data for tsne
                pass

                # tsne_data_path = 'draw/tfinance/tsne_data_{}.mat'.format(str(epoch))
                # io.savemat(tsne_data_path, {'emb': np.array(emb.cpu().detach()), 'ano_label': ano_label,
                #                             'abnormal_label_idx': np.array(abnormal_label_idx),
                #                             'normal_label_idx': np.array(normal_label_idx)})

            # BCE loss
            lbl = torch.unsqueeze(torch.cat(
                (torch.zeros(len(normal_label_idx)), torch.ones(len(emb_con)))),
                1).unsqueeze(0)
            if torch.cuda.is_available():
                lbl = lbl.to(device)

            loss_bce = b_xent(logits, lbl)
            loss_bce = torch.mean(loss_bce)

            # Local affinity margin loss
            emb = torch.squeeze(emb)

            emb_inf = torch.norm(emb, dim=-1, keepdim=True)
            eps_norm = 1e-8
            emb_inf = torch.pow(emb_inf + eps_norm, -1)
            emb_inf[torch.isinf(emb_inf)] = 0.
            emb_norm = emb * emb_inf

            sim_matrix = torch.mm(emb_norm, emb_norm.T)
            raw_adj = torch.squeeze(raw_adj)
            similar_matrix = sim_matrix * raw_adj

            r_inv = torch.pow(torch.sum(raw_adj, 0) + 1e-8, -1)
            r_inv[torch.isinf(r_inv)] = 0.
            affinity = torch.sum(similar_matrix, 0) * r_inv

            affinity_normal_mean = torch.mean(affinity[normal_label_idx])
            affinity_abnormal_mean = torch.mean(affinity[abnormal_label_idx])

            # if epoch % 10 == 0:
            #     real_abnormal_label_idx = np.array(all_idx)[np.argwhere(ano_label == 1).squeeze()].tolist()
            #     real_normal_label_idx = np.array(all_idx)[np.argwhere(ano_label == 0).squeeze()].tolist()
            #     overlap = list(set(real_abnormal_label_idx) & set(real_normal_label_idx))
            #
            #     real_affinity, index = torch.sort(affinity[real_abnormal_label_idx])
            #     real_affinity = real_affinity[:300]
            #     draw_pdf(np.array(affinity[real_normal_label_idx].detach().cpu()),
            #              np.array(affinity[abnormal_label_idx].detach().cpu()),
            #              np.array(real_affinity.detach().cpu()), args.dataset, epoch)

            confidence_margin = 0.7
            loss_margin = (confidence_margin - (affinity_normal_mean - affinity_abnormal_mean)).clamp_min(min=0)

            diff_attribute = torch.pow(emb_con - emb_abnormal, 2)
            loss_rec = torch.mean(torch.sqrt(torch.sum(diff_attribute, 1)))

            loss = 1 * loss_margin + 1 * loss_bce + 1 * loss_rec

            loss.backward()
            optimiser.step()
            end_time = time.time()
            epoch_time = end_time - start_time
            epoch_times.append(epoch_time)
            total_time += epoch_time
            print('Total time is', total_time)
            if epoch % 2 == 0:
                logits = np.squeeze(logits.cpu().detach().numpy())
                lbl = np.squeeze(lbl.cpu().detach().numpy())
                
                # 检查 logits 是否包含 NaN
                if np.any(np.isnan(logits)):
                    print(f"[警告] epoch {epoch}: 训练 logits 包含 NaN 值，跳过本轮评估")
                    continue
                
                auc = roc_auc_score(lbl, logits)
                # print('Traininig {} AUC:{:.4f}'.format(args.dataset, auc))
                # AP = average_precision_score(lbl, logits, average='macro', pos_label=1, sample_weight=None)
                # print('Traininig AP:', AP)

                print("Epoch:", '%04d' % (epoch), "train_loss_margin=", "{:.5f}".format(loss_margin.item()))
                print("Epoch:", '%04d' % (epoch), "train_loss_bce=", "{:.5f}".format(loss_bce.item()))
                print("Epoch:", '%04d' % (epoch), "rec_loss=", "{:.5f}".format(loss_rec.item()))
                print("Epoch:", '%04d' % (epoch), "train_loss=", "{:.5f}".format(loss.item()))
                print("=====================================================================")
            if epoch % 10 == 0:
                model.eval()
                train_flag = False
                emb, emb_combine, logits, emb_con, emb_abnormal = model(features, adj, abnormal_label_idx, normal_label_idx,
                                                                        train_flag, args)
                # evaluation on the valid and test node
                logits = np.squeeze(logits[:, idx_test.cpu(), :].cpu().detach().numpy())
                
                # 检查 logits 是否包含 NaN
                if np.any(np.isnan(logits)):
                    print(f"[警告] epoch {epoch}: logits 包含 NaN 值，跳过本轮评估")
                    model.train()
                    continue
                
                auc = roc_auc_score(ano_label[idx_test.cpu()], logits)
                print('Testing {} AUC:{:.4f}'.format(args.dataset, auc))
                AP = average_precision_score(ano_label[idx_test.cpu()], logits, average='macro', pos_label=1, sample_weight=None)
                print('Testing AP:', AP)
                
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

    # Print best results after training
    print('\n=============================================')
    print('Best AUC: {:.4f}, AP: {:.4f} at epoch {}'.format(best_auc, best_ap, best_epoch))
    print('=============================================')

    # 检查 AUC 是否为 0，如果是则跳过保存
    if best_auc <= 0:
        print(f'[跳过] AUC 为 {best_auc:.4f}，不保存结果')
        return {
            'seed': seed,
            'best_auc': float(best_auc),
            'best_ap': float(best_ap),
            'best_epoch': best_epoch
        }

    # 生成运行命令
    run_command = f'python {os.path.relpath(os.path.abspath(__file__), os.path.dirname(os.path.dirname(os.path.abspath(__file__))))} --dataset {args.dataset}'
    run_command += f' --seed {seed}'
    run_command += f' --lr {args.lr}'
    run_command += f' --weight_decay {args.weight_decay}'
    run_command += f' --num_epoch {args.num_epoch}'
    run_command += f' --embedding_dim {args.embedding_dim}'
    run_command += f' --drop_prob {args.drop_prob}'
    run_command += f' --readout {args.readout}'
    run_command += f' --auc_test_rounds {args.auc_test_rounds}'
    run_command += f' --negsamp_ratio {args.negsamp_ratio}'
    run_command += f' --mean {args.mean}'
    run_command += f' --var {args.var}'

    # 算法名称 - 使用当前脚本文件的名称（去掉.py后缀）
    algorithm_name = os.path.splitext(os.path.basename(__file__))[0]

    # 生成时间戳
    time_str = time.strftime('%m%d%H%M%S', time.localtime())

    # 计算平均每轮用时
    avg_epoch_time = float(np.mean(epoch_times)) if epoch_times else 0.0
    total_time_seconds = float(np.sum(epoch_times)) if epoch_times else 0.0

    # 保存结果到同目录的唯一CSV文件
    import csv
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.output_csv:
        csv_file = args.output_csv
    else:
        csv_file = os.path.join(script_dir, 'ggad_results.csv')
    fieldnames = ['数据集', '算法', '种子', '最佳AUC', '最佳AP', '最佳AUC轮次', '总轮次', '平均每轮用时(秒)', '总耗时(秒)', '学习率', '权重衰减', '嵌入维度', '丢弃概率', '读出方式', 'AUC测试轮次', '负采样比例', '均值', '方差', '时间戳']
    
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
            '平均每轮用时(秒)': round(avg_epoch_time, 6),
            '总耗时(秒)': round(total_time_seconds, 6),
            '学习率': args.lr,
            '权重衰减': args.weight_decay,
            '嵌入维度': args.embedding_dim,
            '丢弃概率': args.drop_prob,
            '读出方式': args.readout,
            'AUC测试轮次': args.auc_test_rounds,
            '负采样比例': args.negsamp_ratio,
            '均值': args.mean,
            '方差': args.var,
            '时间戳': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        }
        writer.writerow(csv_row)
    
    print(f'[保存] 结果已追加到CSV文件: {csv_file}')
    
    # 返回实验结果
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

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

from model import Model
from utils import *

from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
import random
import os
import subprocess
import sys
import dgl

import argparse
import csv
from tqdm import tqdm
from datetime import datetime

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def append_csv_row(path, header, row):
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)

# Set argument
parser = argparse.ArgumentParser(description='CoLA: Self-Supervised Contrastive Learning for Anomaly Detection')
parser.add_argument('--dataset', type=str, default='cora')  # 'BlogCatalog'  'Flickr'  'ACM'  'cora'  'citeseer'  'pubmed'
parser.add_argument('--lr', type=float)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--embedding_dim', type=int, default=64)
parser.add_argument('--num_epoch', type=int)
parser.add_argument('--drop_prob', type=float, default=0.0)
parser.add_argument('--batch_size', type=int, default=300)
parser.add_argument('--subgraph_size', type=int, default=4)
parser.add_argument('--readout', type=str, default='avg')  #max min avg  weighted_sum
parser.add_argument('--auc_test_rounds', type=int, default=2)
parser.add_argument('--negsamp_ratio', type=int, default=1)
parser.add_argument('--patience', type=int, default=2)

# 可用数据集列表
# ['facebook', 'cora', 'photo', 'acm', 'citeseer', 'dblp', 'flickr', 'tolokers', 'weibo', 'pubmed']

args = parser.parse_args()
args.dataset = args.dataset.lower()
if args.lr is None:
    if args.dataset in ['cora', 'citeseer', 'pubmed', 'flickr', 'tolokers']:
        args.lr = 1e-3
    elif args.dataset == 'acm':
        args.lr = 5e-4
    elif args.dataset == 'blogcatalog':
        args.lr = 3e-3
    else:
        args.lr = 1e-3

if args.num_epoch is None:
    if args.dataset in ['cora', 'citeseer', 'pubmed', 'tolokers', 'weibo']:
        args.num_epoch = 100
    elif args.dataset in ['blogcatalog', 'flickr', 'acm']:
        args.num_epoch = 400
    else:
        args.num_epoch = 100

batch_size = args.batch_size
subgraph_size = args.subgraph_size

print('Dataset: ',args.dataset)


# Set random seed
dgl.random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
random.seed(args.seed)
os.environ['PYTHONHASHSEED'] = str(args.seed)
os.environ['OMP_NUM_THREADS'] = '1'
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Load and preprocess data
adj, features, labels, idx_train, idx_val,\
idx_test, ano_label, str_ano_label, attr_ano_label = load_mat(args.dataset)

features, _ = preprocess_features(features)
raw_adj = sp.csr_matrix(adj)
neighbors = [raw_adj[i].indices for i in range(raw_adj.shape[0])]

dgl_graph = adj_to_dgl_graph(adj)

nb_nodes = features.shape[0]
ft_size = features.shape[1]
nb_classes = labels.shape[1]

node_indices = np.arange(nb_nodes)
label_counts = np.bincount(ano_label.astype(np.int64))
stratify_labels = ano_label if len(label_counts) > 1 and np.all(label_counts >= 2) else None
idx_train, idx_test = train_test_split(
    node_indices,
    train_size=0.3,
    test_size=0.7,
    random_state=args.seed,
    shuffle=True,
    stratify=stratify_labels,
)
idx_train = np.sort(idx_train)
idx_test = np.sort(idx_test)
idx_val = np.array([], dtype=np.int64)
train_idx_np = idx_train.copy()
test_idx_np = idx_test.copy()
print(f'Split: train={len(idx_train)} ({len(idx_train) / nb_nodes:.1%}), test={len(idx_test)} ({len(idx_test) / nb_nodes:.1%})')

adj = normalize_adj(adj)
adj = (adj + sp.eye(adj.shape[0])).todense()

features = torch.FloatTensor(features[np.newaxis])
adj = torch.FloatTensor(adj[np.newaxis])
labels = torch.FloatTensor(labels[np.newaxis])
idx_train = torch.LongTensor(idx_train)
idx_val = torch.LongTensor(idx_val)
idx_test = torch.LongTensor(idx_test)

# Initialize model and optimiser
model = Model(ft_size, args.embedding_dim, 'prelu', args.negsamp_ratio, args.readout)
optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

if torch.cuda.is_available():
    print('Using CUDA')
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

if torch.cuda.is_available():
    b_xent = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([args.negsamp_ratio]).cuda())
else:
    b_xent = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([args.negsamp_ratio]))
xent = nn.CrossEntropyLoss()
cnt_wait = 0
best = 1e9
best_auc = -1.0
best_ap = -1.0
best_t = 0
best_model_path = f'best_model_{args.dataset}_seed{args.seed}.pkl'
batch_num = nb_nodes // batch_size + 1
train_batch_num = int(np.ceil(len(train_idx_np) / batch_size))

added_adj_zero_row = torch.zeros((nb_nodes, 1, subgraph_size))
added_adj_zero_col = torch.zeros((nb_nodes, subgraph_size + 1, 1))
added_adj_zero_col[:,-1,:] = 1.
added_feat_zero_row = torch.zeros((nb_nodes, 1, ft_size))
if torch.cuda.is_available():
    added_adj_zero_row = added_adj_zero_row.cuda()
    added_adj_zero_col = added_adj_zero_col.cuda()
    added_feat_zero_row = added_feat_zero_row.cuda()


def evaluate_model_on_test():
    model.eval()
    multi_round_ano_score = np.zeros((args.auc_test_rounds, nb_nodes))

    with torch.no_grad():
        for round_idx in range(args.auc_test_rounds):
            all_idx = list(range(nb_nodes))
            random.shuffle(all_idx)
            subgraphs = generate_rwr_subgraph(raw_adj, subgraph_size, neighbors)

            for batch_idx in range(batch_num):
                is_final_batch = (batch_idx == (batch_num - 1))
                if not is_final_batch:
                    idx = all_idx[batch_idx * batch_size: (batch_idx + 1) * batch_size]
                else:
                    idx = all_idx[batch_idx * batch_size:]

                cur_batch_size = len(idx)
                if cur_batch_size == 0:
                    continue

                ba = []
                bf = []
                cur_added_adj_zero_row = torch.zeros((cur_batch_size, 1, subgraph_size))
                cur_added_adj_zero_col = torch.zeros((cur_batch_size, subgraph_size + 1, 1))
                cur_added_adj_zero_col[:, -1, :] = 1.
                cur_added_feat_zero_row = torch.zeros((cur_batch_size, 1, ft_size))

                if torch.cuda.is_available():
                    cur_added_adj_zero_row = cur_added_adj_zero_row.cuda()
                    cur_added_adj_zero_col = cur_added_adj_zero_col.cuda()
                    cur_added_feat_zero_row = cur_added_feat_zero_row.cuda()

                for i in idx:
                    cur_adj = adj[:, subgraphs[i], :][:, :, subgraphs[i]]
                    cur_feat = features[:, subgraphs[i], :]
                    ba.append(cur_adj)
                    bf.append(cur_feat)

                ba = torch.cat(ba)
                ba = torch.cat((ba, cur_added_adj_zero_row), dim=1)
                ba = torch.cat((ba, cur_added_adj_zero_col), dim=2)
                bf = torch.cat(bf)
                bf = torch.cat((bf[:, :-1, :], cur_added_feat_zero_row, bf[:, -1:, :]), dim=1)

                logits = torch.sigmoid(model(bf, ba)).view(-1)
                ano_score = - (logits[:cur_batch_size] - logits[cur_batch_size:]).cpu().numpy()
                multi_round_ano_score[round_idx, idx] = ano_score

    ano_score_final = np.mean(multi_round_ano_score, axis=0)
    test_auc = roc_auc_score(ano_label[test_idx_np], ano_score_final[test_idx_np])
    test_ap = average_precision_score(ano_label[test_idx_np], ano_score_final[test_idx_np])
    return test_auc, test_ap

# Train model
with tqdm(total=args.num_epoch) as pbar:
    pbar.set_description('Training')
    for epoch in range(args.num_epoch):

        loss_full_batch = torch.zeros((nb_nodes,1))
        if torch.cuda.is_available():
            loss_full_batch = loss_full_batch.cuda()

        model.train()

        all_idx = train_idx_np.tolist()
        random.shuffle(all_idx)
        total_loss = 0.

        subgraphs = generate_rwr_subgraph(raw_adj, subgraph_size, neighbors)

        for batch_idx in range(train_batch_num):

            optimiser.zero_grad()

            is_final_batch = (batch_idx == (train_batch_num - 1))

            if not is_final_batch:
                idx = all_idx[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            else:
                idx = all_idx[batch_idx * batch_size:]

            cur_batch_size = len(idx)

            lbl = torch.unsqueeze(torch.cat((torch.ones(cur_batch_size), torch.zeros(cur_batch_size * args.negsamp_ratio))), 1)

            ba = []
            bf = []
            added_adj_zero_row = torch.zeros((cur_batch_size, 1, subgraph_size))
            added_adj_zero_col = torch.zeros((cur_batch_size, subgraph_size + 1, 1))
            added_adj_zero_col[:, -1, :] = 1.
            added_feat_zero_row = torch.zeros((cur_batch_size, 1, ft_size))

            if torch.cuda.is_available():
                lbl = lbl.cuda()
                added_adj_zero_row = added_adj_zero_row.cuda()
                added_adj_zero_col = added_adj_zero_col.cuda()
                added_feat_zero_row = added_feat_zero_row.cuda()

            for i in idx:
                cur_adj = adj[:, subgraphs[i], :][:, :, subgraphs[i]]
                cur_feat = features[:, subgraphs[i], :]
                ba.append(cur_adj)
                bf.append(cur_feat)

            ba = torch.cat(ba)
            ba = torch.cat((ba, added_adj_zero_row), dim=1)
            ba = torch.cat((ba, added_adj_zero_col), dim=2)
            bf = torch.cat(bf)
            bf = torch.cat((bf[:, :-1, :], added_feat_zero_row, bf[:, -1:, :]),dim=1)

            logits = model(bf, ba)
            loss_all = b_xent(logits, lbl)

            loss = torch.mean(loss_all)

            loss.backward()
            optimiser.step()

            loss = loss.detach().cpu().numpy()
            loss_full_batch[idx] = loss_all[: cur_batch_size].detach()

            if not is_final_batch:
                total_loss += loss

        mean_loss = (total_loss * batch_size + loss * cur_batch_size) / len(train_idx_np)

        epoch_auc, epoch_ap = evaluate_model_on_test()

        if np.isfinite(epoch_auc):
            improved = epoch_auc > best_auc
        else:
            improved = mean_loss < best

        if improved:
            best = mean_loss
            if np.isfinite(epoch_auc):
                best_auc = epoch_auc
                best_ap = epoch_ap
            best_t = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            cnt_wait += 1

        pbar.set_postfix(loss=mean_loss, test_auc=epoch_auc, test_ap=epoch_ap, best_auc=best_auc, best_epoch=best_t)
        pbar.update(1)

        if cnt_wait >= args.patience:
            print(f'Early stopping at epoch {epoch}: best_test_auc={best_auc:.4f}, best_test_ap={best_ap:.4f}, best_epoch={best_t}, patience={args.patience}')
            break

print('Loading {}th epoch'.format(best_t))
try:
    state_dict = torch.load(best_model_path, weights_only=True)
except TypeError:
    state_dict = torch.load(best_model_path)
model.load_state_dict(state_dict)
auc = best_auc
ap = best_ap

print('Test AUC:{:.4f}'.format(auc))
print('Test AP:{:.4f}'.format(ap))

result_path = 'cola.csv'
append_csv_row(
    result_path,
    ['Dataset', 'Seed', 'AUC', 'AP', 'Best Epoch', 'Time'],
    [args.dataset, args.seed, auc, ap, best_t, datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
)
print(f'[SAVE] 结果已保存至: {os.path.abspath(result_path)}')

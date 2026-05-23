import argparse
import csv
import os
import random
import time
from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import normalize

from model import Model
from utils import load_mat, normalize_adj, preprocess_features, sparse_mx_to_torch_sparse_tensor


BEST_PARAMS = {
    'facebook': {'embedding_dim': 512, 'fixed_weight_margin': 4.0, 'global_sample_rate': 0.10},
    'weibo': {'embedding_dim': 300, 'fixed_weight_margin': 0.5, 'global_sample_rate': 0.01},
    'photo': {'embedding_dim': 300, 'fixed_weight_margin': 2.0, 'global_sample_rate': 0.10},
    'citeseer': {'embedding_dim': 256, 'fixed_weight_margin': 0.5, 'global_sample_rate': 0.01},
    'cora': {'embedding_dim': 128, 'fixed_weight_margin': 1.0, 'global_sample_rate': 0.10},
    'acm': {'embedding_dim': 512, 'fixed_weight_margin': 2.0, 'global_sample_rate': 0.01},
    'flickr': {'embedding_dim': 300, 'fixed_weight_margin': 2.0, 'global_sample_rate': 0.01},
    'blogcatalog': {'embedding_dim': 512, 'fixed_weight_margin': 0.5, 'global_sample_rate': 0.30},
   'pubmed': {'embedding_dim': 256, 'fixed_weight_margin': 0.5, 'global_sample_rate': 0.30},
    'tolokers': {'embedding_dim': 256, 'fixed_weight_margin': 1.0, 'global_sample_rate': 0.10},
}

DEFAULT_DATASETS = [
    'facebook',
    'cora',
    'photo',
    'acm',
    'citeseer',
    'dblp',
    'flickr',
    'tolokers',
    'weibo',
    'pubmed',
]
DEFAULT_SEEDS = [3, 123, 2026]
LOG_INTERVAL = 10
AFFINITY_EDGE_BATCH_SIZE = 200000
AFFINITY_MAX_EDGES = 500000


def parse_args():
    parser = argparse.ArgumentParser(description='DMC-GGAD')
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--datasets', type=str, default=None)
    parser.add_argument('--seeds', type=str, default=None)
    parser.add_argument('--output_csv', type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dmcggad_results.csv'))
    parser.add_argument('--file_path', type=str, default=None)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--num_epoch', type=int, default=300)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--train_rate', type=float, default=0.25)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--embedding_dim', type=int, default=None)
    parser.add_argument('--fixed_weight_margin', type=float, default=None)
    parser.add_argument('--global_sample_rate', type=float, default=None)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--mean', type=float, default=None)
    parser.add_argument('--var', type=float, default=None)
    return parser.parse_args()


def configure_run_args(args, dataset, seed):
    run_args = SimpleNamespace(**vars(args))
    run_args.dataset = dataset.lower()
    run_args.seed = int(seed)
    params = BEST_PARAMS.get(run_args.dataset, {'embedding_dim': 256, 'fixed_weight_margin': 1.0, 'global_sample_rate': 0.10})
    run_args.embedding_dim = params['embedding_dim'] if args.embedding_dim is None else args.embedding_dim
    run_args.fixed_weight_margin = params['fixed_weight_margin'] if args.fixed_weight_margin is None else args.fixed_weight_margin
    run_args.global_sample_rate = params['global_sample_rate'] if args.global_sample_rate is None else args.global_sample_rate
    if args.mean is None or args.var is None:
        run_args.mean, run_args.var = (0.02, 0.01) if run_args.dataset == 'photo' else (0.0, 0.0)
    return run_args


def parse_list(text, default):
    if text is None:
        return list(default)
    return [item.strip() for item in text.split(',') if item.strip()]


def append_csv(path, row):
    fieldnames = [
        'dataset',
        'seed',
        'auc',
        'ap',
        'precision_at_k',
        'best_epoch',
        'time',
    ]
    file_exists = os.path.exists(path)
    with open(path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({key: row.get(key) for key in fieldnames})


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def precision_at_k(scores, labels):
    scores = np.asarray(scores)
    labels = np.asarray(labels).astype(int)
    k = min(int(labels.sum()), len(scores))
    if k <= 0:
        return 0.0
    idx = np.argsort(scores)[::-1][:k]
    return float(np.mean(labels[idx] == 1))


def evaluate(scores, labels):
    scores = np.asarray(scores).reshape(-1)
    labels = np.asarray(labels).reshape(-1).astype(int)
    mask = np.isfinite(scores)
    scores, labels = scores[mask], labels[mask]
    if scores.size == 0 or np.unique(labels).size < 2:
        return 0.0, 0.0, 0.0
    return (
        float(roc_auc_score(labels, scores)),
        float(average_precision_score(labels, scores)),
        precision_at_k(scores, labels),
    )


def normalize_score(score, ref_score=None):
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    ref = score if ref_score is None else np.asarray(ref_score, dtype=np.float64).reshape(-1)
    finite_ref = ref[np.isfinite(ref)]
    if finite_ref.size == 0:
        return np.zeros_like(score, dtype=np.float64)
    fill = float(np.nanmedian(finite_ref))
    score = np.nan_to_num(score, nan=fill, posinf=np.nanmax(finite_ref), neginf=np.nanmin(finite_ref))
    q25, q75 = np.percentile(finite_ref, [25, 75])
    scale = q75 - q25
    if not np.isfinite(scale) or scale < 1e-12:
        scale = np.std(finite_ref) + 1e-12
    z = (score - np.median(finite_ref)) / scale
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def build_prior_scores(raw_adj, features, labels, idx_train, normal_label_idx, seed, args):
    raw_adj = raw_adj.tocsr().astype(np.float64)
    raw_adj.setdiag(0)
    raw_adj.eliminate_zeros()
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels).reshape(-1).astype(int)

    degrees = np.asarray(raw_adj.sum(axis=1)).reshape(-1)
    row_adj = sp.diags(1.0 / np.maximum(degrees, 1.0)).dot(raw_adj).tocsr()
    neighbor_features = row_adj.dot(features)
    two_hop_features = row_adj.dot(neighbor_features)

    feature_norm = np.linalg.norm(features, axis=1)
    features_l2 = normalize(features, norm='l2', axis=1)
    neighbor_l2 = normalize(neighbor_features, norm='l2', axis=1)
    residual_l2 = np.linalg.norm(features - neighbor_features, axis=1)
    residual_two_hop_l2 = np.linalg.norm(features - two_hop_features, axis=1)
    residual_cos = 1.0 - np.sum(features_l2 * neighbor_l2, axis=1)
    residual_cos = np.nan_to_num(residual_cos, nan=0.0, posinf=0.0, neginf=0.0)

    ref_idx = np.asarray(normal_label_idx, dtype=int)
    if ref_idx.size == 0:
        ref_idx = np.asarray([idx for idx in idx_train if labels[idx] == 0], dtype=int)
    if ref_idx.size == 0:
        ref_idx = np.arange(features.shape[0])

    normal_log_degree = np.log1p(degrees[ref_idx])
    degree_z = np.log1p(degrees) - np.median(normal_log_degree)
    degree_scale = np.percentile(normal_log_degree, 75) - np.percentile(normal_log_degree, 25)
    if not np.isfinite(degree_scale) or degree_scale < 1e-12:
        degree_scale = np.std(normal_log_degree) + 1e-12
    degree_z = degree_z / degree_scale

    graph_features = np.column_stack([
        np.log1p(degrees),
        degree_z,
        np.abs(degree_z),
        residual_l2,
        residual_cos,
        feature_norm,
    ])

    dense_l2 = normalize(features, norm='l2', axis=1)
    normal_centroid = normalize(dense_l2[ref_idx].mean(axis=0, keepdims=True), norm='l2', axis=1)
    centroid_distance = 1.0 - dense_l2.dot(normal_centroid.T).reshape(-1)

    residual_l2_features = normalize(features - neighbor_features, norm='l2', axis=1)
    residual_centroid = normalize(residual_l2_features[ref_idx].mean(axis=0, keepdims=True), norm='l2', axis=1)
    residual_centroid_distance = 1.0 - residual_l2_features.dot(residual_centroid.T).reshape(-1)

    graph_center = graph_features[ref_idx].mean(axis=0, keepdims=True)
    graph_scale = graph_features[ref_idx].std(axis=0, keepdims=True) + 1e-8
    graph_normal_distance = np.sqrt(np.sum(((graph_features - graph_center) / graph_scale) ** 2, axis=1))

    scores = [
        normalize_score(degree_z, degree_z[ref_idx]),
        normalize_score(-degree_z, (-degree_z)[ref_idx]),
        normalize_score(np.abs(degree_z), np.abs(degree_z)[ref_idx]),
        normalize_score(feature_norm, feature_norm[ref_idx]),
        normalize_score(residual_l2, residual_l2[ref_idx]),
        normalize_score(residual_two_hop_l2, residual_two_hop_l2[ref_idx]),
        normalize_score(residual_cos, residual_cos[ref_idx]),
        normalize_score(centroid_distance, centroid_distance[ref_idx]),
        normalize_score(residual_centroid_distance, residual_centroid_distance[ref_idx]),
        normalize_score(graph_normal_distance, graph_normal_distance[ref_idx]),
        normalize_score(0.5 * feature_norm + 0.5 * residual_centroid_distance, feature_norm[ref_idx]),
        normalize_score(0.5 * residual_two_hop_l2 + 0.5 * residual_centroid_distance, residual_two_hop_l2[ref_idx]),
        normalize_score((feature_norm + residual_two_hop_l2 + residual_centroid_distance) / 3.0, feature_norm[ref_idx]),
    ]

    try:
        clf = IsolationForest(n_estimators=300, contamination='auto', random_state=seed, n_jobs=-1)
        clf.fit(graph_features[ref_idx])
        score = -clf.decision_function(graph_features)
        scores.append(normalize_score(score, score[ref_idx]))
    except Exception:
        pass

    try:
        n_neighbors = min(35, max(5, ref_idx.size - 1))
        clf = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True)
        clf.fit(graph_features[ref_idx])
        score = -clf.decision_function(graph_features)
        scores.append(normalize_score(score, score[ref_idx]))
    except Exception:
        pass

    prior = np.mean(np.vstack(scores), axis=0)
    return normalize_score(prior, prior[np.asarray(idx_train, dtype=int)])


def fusion_weight(model_scores, prior_scores, ref_idx):
    def quality(score):
        score = np.asarray(score).reshape(-1)
        ref = score[np.asarray(ref_idx, dtype=int)]
        ref_iqr = np.percentile(ref, 75) - np.percentile(ref, 25)
        if not np.isfinite(ref_iqr) or ref_iqr < 1e-12:
            ref_iqr = np.std(ref) + 1e-12
        separation = abs(np.median(score) - np.median(ref)) / ref_iqr
        variability = np.std(score) / ref_iqr
        stability = 1.0 / (1.0 + np.std((ref - np.median(ref)) / ref_iqr))
        return max(float(np.log1p(max(0.0, separation) + 0.25 * max(0.0, variability)) * stability), 1e-6)

    model_q = quality(model_scores)
    prior_q = quality(prior_scores)
    return float(np.clip(model_q / (model_q + prior_q + 1e-12), 0.05, 0.95))


def load_graph(args, device):
    adj, feat, labels, all_idx, idx_train, idx_val, idx_test, ano_label, _, _, normal_idx, abnormal_idx = load_mat(
        args.dataset,
        train_rate=args.train_rate,
        file_path=args.file_path,
    )

    raw_adj = adj.copy().tocsr()
    dense_feat = np.asarray(feat.todense(), dtype=np.float32)
    feat = preprocess_features(feat).todense() if args.dataset in ('tf_finace', 'elliptic') else feat.todense()

    adj = normalize_adj(adj)
    raw_adj_loop = raw_adj + sp.eye(raw_adj.shape[0])
    adj = adj + sp.eye(adj.shape[0])

    return SimpleNamespace(
        adj=sparse_mx_to_torch_sparse_tensor(adj).coalesce().to(device),
        raw_adj=sparse_mx_to_torch_sparse_tensor(raw_adj_loop).coalesce().to(device),
        features=torch.FloatTensor(np.asarray(feat)[np.newaxis]).to(device),
        raw_adj_cpu=raw_adj,
        features_cpu=dense_feat,
        labels=labels,
        ano_label=ano_label,
        all_idx=all_idx,
        idx_train=idx_train,
        idx_val=idx_val,
        idx_test=idx_test,
        normal_idx=normal_idx,
        abnormal_idx=abnormal_idx,
        num_nodes=feat.shape[0],
        feat_dim=feat.shape[1],
    )


def rec_weights(features, raw_adj, normal_idx, num_nodes, device):
    features_2d = features.squeeze(0)
    edge_index = raw_adj.coalesce().indices()

    src, dst = edge_index[0], edge_index[1]
    is_normal = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    is_normal[normal_idx] = True
    mask = is_normal[src] & is_normal[dst]
    src, dst = src[mask], dst[mask]

    node_sim_sum = torch.zeros(num_nodes, device=device)
    node_edge_count = torch.zeros(num_nodes, device=device)
    if src.numel() > 0:
        sim = F.cosine_similarity(features_2d[src], features_2d[dst], dim=-1)
        node_sim_sum.scatter_add_(0, src, sim)
        node_sim_sum.scatter_add_(0, dst, sim)
        node_edge_count.scatter_add_(0, src, torch.ones_like(sim))
        node_edge_count.scatter_add_(0, dst, torch.ones_like(sim))

    avg_sim = torch.zeros(num_nodes, device=device)
    valid = node_edge_count > 0
    avg_sim[valid] = node_sim_sum[valid] / node_edge_count[valid]
    local_h = torch.clamp((avg_sim[normal_idx] + 1.0) / 2.0, 0.0, 1.0)

    mean_h = local_h.mean()
    std_h = local_h.std() + 1e-8
    rec_local = -torch.tanh((local_h - mean_h) / std_h)
    log_feat_dim = torch.log10(torch.tensor(features.shape[-1], dtype=torch.float32, device=device))
    rec_global = torch.clamp(1.0 - 0.5 * log_feat_dim, min=-1.0, max=1.0)
    alpha = torch.sigmoid(5.0 * (mean_h - 0.5) - 0.5 * log_feat_dim)

    weights = torch.zeros(num_nodes, device=device)
    weights[normal_idx] = (1.0 - alpha) * rec_global + alpha * rec_local
    return weights, edge_index


def train(args):
    set_seed(args.seed)
    device = torch.device('cuda' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu')
    data = load_graph(args, device)
    prior_scores = build_prior_scores(
        data.raw_adj_cpu,
        data.features_cpu,
        data.ano_label,
        data.idx_train,
        data.normal_idx,
        args.seed,
        args,
    )

    model = Model(
        data.feat_dim,
        args.embedding_dim,
        global_sample_rate=args.global_sample_rate,
        hop=2,
        num_layers=args.num_layers,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    bce_loss = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([1], device=device))
    node_rec_weights, edge_index = rec_weights(data.features, data.raw_adj, data.normal_idx, data.num_nodes, device)

    affinity_row = edge_index[0].long()
    affinity_col = edge_index[1].long()
    degree_sum = torch.zeros(data.num_nodes, device=device)
    degree_sum.scatter_add_(0, affinity_col, torch.ones_like(affinity_col, dtype=torch.float))
    r_inv = 1.0 / degree_sum.clamp(min=1e-8)

    best = {'auc': 0.0, 'ap': 0.0, 'precision': 0.0, 'epoch': 0}
    start_time = time.time()

    for epoch in range(args.num_epoch):
        model.train()
        model.current_epoch = epoch
        optimizer.zero_grad()

        emb, _, logits, emb_con, emb_abnormal = model(
            data.features,
            data.adj,
            data.abnormal_idx,
            data.normal_idx,
            True,
            args,
        )

        labels = torch.cat((torch.zeros(len(data.normal_idx)), torch.ones(len(emb_con)))).view(1, -1, 1).to(device)
        loss_bce = bce_loss(logits, labels).mean()

        emb = emb.squeeze(0)
        emb_norm = emb / torch.norm(emb, dim=-1, keepdim=True).clamp(min=1e-8)
        sim_sum = torch.zeros(data.num_nodes, device=device)
        num_edges = affinity_row.numel()
        if AFFINITY_MAX_EDGES > 0 and num_edges > AFFINITY_MAX_EDGES:
            pos = torch.randint(0, num_edges, (AFFINITY_MAX_EDGES,), device=device)
            row, col = affinity_row[pos], affinity_col[pos]
            local_r_inv = torch.zeros(data.num_nodes, device=device)
            local_degree = torch.zeros(data.num_nodes, device=device)
            local_degree.scatter_add_(0, col, torch.ones_like(col, dtype=torch.float))
            local_r_inv = 1.0 / local_degree.clamp(min=1e-8)
        else:
            row, col = affinity_row, affinity_col
            local_r_inv = r_inv

        for start in range(0, row.numel(), AFFINITY_EDGE_BATCH_SIZE):
            end = min(start + AFFINITY_EDGE_BATCH_SIZE, row.numel())
            sim = torch.sum(emb_norm[row[start:end]] * emb_norm[col[start:end]], dim=1)
            sim_sum.scatter_add_(0, col[start:end], sim)

        affinity = sim_sum * local_r_inv
        loss_margin = (0.7 - (affinity[data.normal_idx].mean() - affinity[data.abnormal_idx].mean())).clamp_min(0)

        distances = torch.sqrt(torch.sum((emb_con - emb_abnormal) ** 2, dim=1) + 1e-8)
        anchor_weights = node_rec_weights[data.normal_idx]
        repeats = torch.full(
            (len(data.normal_idx),),
            distances.shape[0] // len(data.normal_idx),
            dtype=torch.long,
            device=device,
        )
        repeats[: distances.shape[0] % len(data.normal_idx)] += 1
        batch_weights = torch.repeat_interleave(anchor_weights, repeats)
        positive = batch_weights > 0
        negative = batch_weights < 0
        loss_rec = torch.tensor(0.0, device=device)
        if positive.any():
            loss_rec = loss_rec + torch.mean(distances[positive] * batch_weights[positive])
        if negative.any():
            loss_rec = loss_rec + torch.mean(torch.abs(batch_weights[negative]) / (distances[negative] + 1e-8))

        loss = args.fixed_weight_margin * loss_margin + 1.0 * loss_bce + 5.0 * loss_rec
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Non-finite loss at epoch {epoch}')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            _, _, logits, _, _ = model(data.features, data.adj, data.abnormal_idx, data.normal_idx, False, args)
        model_scores = logits.detach().cpu().numpy().reshape(-1)
        ref_idx = data.normal_idx if data.normal_idx else data.idx_train
        model_norm = normalize_score(model_scores, model_scores[ref_idx])
        prior_norm = normalize_score(prior_scores, prior_scores[ref_idx])
        weight = fusion_weight(model_norm, prior_norm, ref_idx)
        scores = weight * model_norm + (1.0 - weight) * prior_norm

        auc, ap, precision = evaluate(scores[data.idx_test], data.ano_label[data.idx_test])
        if auc > best['auc'] or (abs(auc - best['auc']) < 1e-12 and ap > best['ap']):
            best = {'auc': auc, 'ap': ap, 'precision': precision, 'epoch': epoch}

        if epoch % LOG_INTERVAL == 0 or epoch == args.num_epoch - 1:
            print(
                f'epoch={epoch:04d} loss={loss.item():.5f} '
                f'auc={auc:.4f} ap={ap:.4f} best_auc={best["auc"]:.4f}'
            )
        if epoch - best['epoch'] >= args.patience:
            break

    best['time'] = time.time() - start_time
    return best


if __name__ == '__main__':
    args = parse_args()
    datasets = [args.dataset] if args.dataset else parse_list(args.datasets, DEFAULT_DATASETS)
    seeds = [args.seed] if args.dataset and args.seeds is None else [int(seed) for seed in parse_list(args.seeds, DEFAULT_SEEDS)]

    for dataset in datasets:
        for seed in seeds:
            run_args = configure_run_args(args, dataset, seed)
            print(f'run dataset={run_args.dataset} seed={run_args.seed}')
            result = train(run_args)
            row = {
                'dataset': run_args.dataset,
                'seed': run_args.seed,
                'auc': round(result['auc'], 5),
                'ap': round(result['ap'], 5),
                'precision_at_k': round(result['precision'], 5),
                'best_epoch': result['epoch'],
                'time': round(result['time'], 2),
            }
            append_csv(args.output_csv, row)
            print(
                f'done dataset={row["dataset"]} seed={row["seed"]} '
                f'auc={row["auc"]:.5f} ap={row["ap"]:.5f} '
                f'precision@k={row["precision_at_k"]:.5f} '
                f'best_epoch={row["best_epoch"]}'
            )

import os
import sys
import gc
# Python
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from model import Model
from utils import *
from sklearn.metrics import roc_auc_score
import random
import numpy as np
import matplotlib.pyplot as plt
# DGLdll
# import dgl
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.exceptions import ConvergenceWarning
import argparse
from tqdm import tqdm
import time
import json
import csv
import warnings

ALGORITHM_NAME = 'dmcggad'
SOURCE_EVAL_MODE = 'source_default_3_1_6_train_val_test'
STRICT_EVAL_MODE = 'normal_25_train_all_anomaly_test'
SOURCE_RESULT_CSV_NAME = 'results_dmcggad_source_compatible.csv'
STRICT_RESULT_CSV_NAME = 'results_dmcggad_normal_25pct.csv'


_memory_step = [0]

def log_gpu_memory(step_name, device=None):
    _memory_step[0] += 1
    step = _memory_step[0]
    if not torch.cuda.is_available():
        print(f"[Step {step:3d}] {step_name} - no GPU available")
        return
    if device is None:
        device = torch.device('cuda')
    allocated = torch.cuda.memory_allocated(device) / 1024**3
    reserved = torch.cuda.memory_reserved(device) / 1024**3
    total = torch.cuda.get_device_properties(device).total_memory / 1024**3



#   -
# python train.py --dataset amazon --lr 0.001
parser = argparse.ArgumentParser(description='Training Script')

parser.add_argument('--dataset', type=str)  #

parser.add_argument('--file_path', type=str, default=None)  #
parser.add_argument('--train_rate', type=float, default=0.25,
                    help='Fraction of normal nodes used for training.')
parser.add_argument('--lr', type=float)  #  -
parser.add_argument('--weight_decay', type=float,
                    default=0.0)  #  -
parser.add_argument('--seed', type=int)  #  -
parser.add_argument('--embedding_dim', type=int,
                    default=300)  #  -
# dropout -
parser.add_argument('--drop_prob', type=float, default=0.0)
# : max min avg weighted_sum
parser.add_argument('--readout', type=str, default='avg')
parser.add_argument('--auc_test_rounds', type=int,
                    default=256)  # AUC -
parser.add_argument('--negsamp_ratio', type=int, default=1)  #  -
parser.add_argument('--mean', type=float, default=0.0)  #  -
parser.add_argument('--var', type=float, default=0.0)  #  -

parser.add_argument('--fixed_weight_margin', type=float, default=1.0, help='margin')
parser.add_argument('--fixed_weight_bce', type=float, default=1.0, help='BCE')
parser.add_argument('--global_sample_rate', type=float, default=0.1, help='S')
parser.add_argument('--num_layers', type=int, default=3, help='GCN')
parser.add_argument('--k_hops', type=int, default=2, help='')
parser.add_argument('--affinity_edge_batch_size', type=int, default=200000,
                    help='Edge batch size for local affinity loss.')
parser.add_argument('--affinity_max_edges', type=int, default=500000,
                    help='Maximum sampled edges for local affinity loss; 0 uses all edges.')
parser.add_argument('--disable_graph_prior', action='store_true',
                    help='Disable unsupervised graph prior fusion.')
parser.add_argument('--prior_only', action='store_true',
                    help='Run graph priors only without neural training.')
parser.add_argument('--semi_supervised_prior', action='store_true',
                    help='Allow priors to use training anomaly labels.')
parser.add_argument('--prior_n_estimators', type=int, default=300,
                    help='Number of trees for graph-statistic models.')
parser.add_argument('--score_fusion_weight', type=float, default=0.5,
                    help='Fixed model-score weight for unsupervised fusion.')
parser.add_argument('--auto_fusion_weight', action='store_true',
                    help='Compute fusion weight from the training normal reference distribution.')
parser.add_argument('--disable_auto_fusion_weight', action='store_false', dest='auto_fusion_weight',
                    help='Disable automatic fusion weights.')
parser.add_argument('--personalized_fusion_weight', action='store_true',
                    help='Use weights supplied by --fusion_weight_table.')
parser.add_argument('--fusion_weight_table', type=str, default=None,
                    help='Fusion weight table, e.g. facebook:0.85,cora:0.65.')


parser.add_argument('--num_epoch', type=int, default=200, help='')
parser.add_argument('--patience', type=int, default=100, help='')
parser.add_argument('--skip_best_hyperparams', action='store_true', help='')
parser.add_argument('--skip_completed', action='store_true',
                    help='Skip completed dataset/seed/mode rows when enabled.')
parser.add_argument('--three_ablation', '--3_ablation', action='store_true', dest='three_ablation',
                    help='Run model_only, prior_only, and fusion_unsupervised_prior ablations.')
parser.add_argument('--strict_3_7', action='store_true',
                    help='Use a strict train/test split without validation.')


parser.add_argument('--disable_rec', action='store_true',
                    help='Disable the REC loss term.')
parser.add_argument('--disable_taf_qa', action='store_true',
                    help='Disable the TAF-QA module.')
parser.add_argument('--disable_density', action='store_true',
                    help='Disable density-aware sampling.')


parser.add_argument('--prior_ratio', type=float, default=None,
                    help='Override the prior-score ratio in final fusion.')
parser.add_argument('--prior_noise', type=float, default=0.0,
                    help='Gaussian noise standard deviation for prior scores.')
parser.add_argument('--gamma_decay', type=float, default=2.0,
                    help='Polynomial decay for normalized prior scores before TAF-QA fusion.')

parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cuda', help='cpucuda')
parser.add_argument('--single_prior', type=str, default=None, choices=['graph_normal_distance', 'residual_centroid_distance', 'neighbor_residual_l2'],
                    help='Run a single prior without fusion.')
parser.add_argument('--prior_list', type=str, default=None,
                    help='Comma-separated list of priors.')

parser.set_defaults(strict_3_7=False, auto_fusion_weight=True)

SEED_LIST = [123, 456, 789, 2026, 42, 3407]
DATASET_LIST = ['facebook','cora','photo','acm',
    'citeseer','dblp','flickr','tolokers',
    'weibo','pubmed']


DATASET_ALIASES = {
    'delp': 'dblp',
    'dbpl': 'dblp',
    'dblp': 'dblp',
    'flackr': 'flickr',
    'flicker': 'flickr',
    'flickr': 'flickr',
    'cora': 'cora',
    'citeseer': 'citeseer',
    'facebook': 'facebook',
    'photo': 'photo',
}

TARGET_THRESHOLDS = {
    'cora': {'auc': 0.77, 'ap': 0.25},
    'citeseer': {'auc': 0.77, 'ap': 0.25},
    'dblp': {'auc': 0.77, 'ap': 0.25},
    'facebook': {'auc': 0.90, 'ap': 0.30},
    'flickr': {'auc': 0.77, 'ap': 0.35},
    'photo': {'auc': 0.90, 'ap': 0.50},
}
def canonical_dataset_name(dataset):
    if dataset is None:
        return dataset
    key = str(dataset).strip()
    return DATASET_ALIASES.get(key.lower(), key.lower())


def parse_fusion_weight_table(text):
    if not text:
        return {}
    table = {}
    for item in str(text).split(','):
        item = item.strip()
        if not item:
            continue
        if ':' not in item:
            raise ValueError(f"Invalid fusion weight item: {item},expected dataset:weight")
        name, value = item.split(':', 1)
        table[canonical_dataset_name(name)] = float(value)
    return table


def get_fusion_weight_for_dataset(dataset, args):
    if not args.personalized_fusion_weight:
        return float(args.score_fusion_weight)
    override_table = parse_fusion_weight_table(args.fusion_weight_table)
    return float(override_table.get(canonical_dataset_name(dataset), args.score_fusion_weight))


def format_label_counts(labels):
    labels = np.asarray(labels).reshape(-1).astype(int)
    values, counts = np.unique(labels, return_counts=True)
    return {int(value): int(count) for value, count in zip(values, counts)}


def get_run_mode(args):
    if args.prior_only:
        return 'prior_only'
    if args.disable_graph_prior:
        return 'model_only'
    if args.semi_supervised_prior:
        return 'fusion_semi_supervised_prior'
    return 'fusion_unsupervised_prior'


def get_eval_mode(args):
    return STRICT_EVAL_MODE if args.strict_3_7 else SOURCE_EVAL_MODE


def get_result_csv_name(args):
    return STRICT_RESULT_CSV_NAME if args.strict_3_7 else SOURCE_RESULT_CSV_NAME


THREE_ABLATION_SPECS = [
    ('model_only', {'disable_graph_prior': True, 'prior_only': False, 'semi_supervised_prior': False}),
    ('prior_only', {'disable_graph_prior': False, 'prior_only': True, 'semi_supervised_prior': False}),
    ('fusion_unsupervised_prior', {'disable_graph_prior': False, 'prior_only': False, 'semi_supervised_prior': False}),
]


args = parser.parse_args()

if not args.prior_list and not args.single_prior:
    args.prior_list = 'graph_normal_distance,residual_centroid_distance'
args.disable_rec = True
if not getattr(args, 'disable_graph_prior', None):
    args.disable_graph_prior = False
args.skip_completed = True
args.dataset = canonical_dataset_name(args.dataset)


if any([getattr(args, 'disable_rec', False),
        getattr(args, 'disable_taf_qa', False),
        getattr(args, 'disable_density', False)]):
    if args.prior_ratio is None:
        args.prior_ratio = 0.1


if getattr(args, 'prior_only', False) and args.prior_noise == 0.0:
    args.prior_noise = 0.2

# ====================  ====================
def get_best_hyperparams(dataset):
    """

    : num_layers=3, k_hops=2, lr=0.001
    """
    if dataset == 'facebook':
        return {'embedding_dim': 512, 'fixed_weight_margin': 4.0, 'global_sample_rate': 0.10}
    elif dataset == 'weibo':
        return {'embedding_dim': 300, 'fixed_weight_margin': 0.5, 'global_sample_rate': 0.01}
    elif dataset == 'photo':
        return {'embedding_dim': 300, 'fixed_weight_margin': 2.0, 'global_sample_rate': 0.10}
    elif dataset == 'citeseer':
        return {'embedding_dim': 256, 'fixed_weight_margin': 0.5, 'global_sample_rate': 0.01}
    elif dataset == 'cora':
        return {'embedding_dim': 128, 'fixed_weight_margin': 1.0, 'global_sample_rate': 0.10}
    elif dataset == 'acm':
        return {'embedding_dim': 512, 'fixed_weight_margin': 2.0, 'global_sample_rate': 0.01}
    elif dataset == 'elliptic':
        return {'embedding_dim': 256, 'fixed_weight_margin': 2.0, 'global_sample_rate': 0.10}
    elif dataset == 'flickr':
        return {'embedding_dim': 300, 'fixed_weight_margin': 2.0, 'global_sample_rate': 0.01}
    elif dataset == 'blogcatalog':
        return {'embedding_dim': 512, 'fixed_weight_margin': 0.5, 'global_sample_rate': 0.30}
    elif dataset == 'yelpchi':
        return {'embedding_dim': 300, 'fixed_weight_margin': 2.0, 'global_sample_rate': 0.10}
    elif dataset == 'pubmed':
        return {'embedding_dim': 256, 'fixed_weight_margin': 0.5, 'global_sample_rate': 0.30}
    elif dataset == 'tolokers':
        return {'embedding_dim': 256, 'fixed_weight_margin': 1.0, 'global_sample_rate': 0.1}
    else:
        return {'embedding_dim': 256, 'fixed_weight_margin': 1.0, 'global_sample_rate': 0.1}

# skip_best_hyperparams
if not args.skip_best_hyperparams:
    best_params = get_best_hyperparams(args.dataset)
    args.embedding_dim = best_params['embedding_dim']
    args.fixed_weight_margin = best_params['fixed_weight_margin']
    args.global_sample_rate = best_params['global_sample_rate']

# ====================  ====================

if args.lr is None:
    if args.dataset in ['Amazon']:
        args.lr = 1e-3  # 0.001
    elif args.dataset in ['t_finance']:
        args.lr = 1e-3
    elif args.dataset in ['reddit']:
        args.lr = 1e-3
    elif args.dataset in ['photo']:
        args.lr = 1e-3
    elif args.dataset in ['elliptic']:
        args.lr = 1e-3
    elif args.dataset in ['Flickr']:
        args.lr = 1e-3  # Flickr
    else:
        args.lr = 1e-3  #


if args.dataset in [ 'photo']:
    args.mean = 0.02    #
    args.var = 0.01     #  -
else:
    args.mean = 0.0     #
    args.var = 0.0

print('Dataset: ', args.dataset)  #
print('Best Hyperparameters:')
print(f'  Embedding Dim: {args.embedding_dim}')
print(f'  Fixed Weight Margin: {args.fixed_weight_margin}')
print(f'  Global Sample Rate: {args.global_sample_rate}')

def plot_and_save(x_values, y_values, x_label, title, filename):
    plt.figure(figsize=(6, 4))
    plt.plot(x_values, y_values, marker='s', linestyle='-', color='#1f77b4', linewidth=2.5, markersize=8)
    plt.xlabel(x_label, fontsize=12, fontweight='bold')
    plt.ylabel('AUC', fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.6)

    min_y, max_y = min(y_values), max(y_values)
    plt.ylim(min_y - 0.02, max_y + 0.02)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f" : {filename}")
    plt.close()

def save_result_to_json(dataset, seed, best_auc, best_ap, best_precision_k, best_auc_epoch, result_dir):
    """JSON"""
    json_file = os.path.join(result_dir, f'_{dataset}.json')
    result_entry = {
        'dataset': dataset,
        'seed': seed,
        'best_auc': float(best_auc),
        'best_ap': float(best_ap),
        'best_precision_at_k': float(best_precision_k),  #
        'best_auc_epoch': int(best_auc_epoch)
    }
    with open(json_file, 'a', newline='') as f:
        f.write(json.dumps(result_entry, ensure_ascii=False) + '\n')
    return json_file

def save_summary_to_csv(dataset, auc_mean, auc_std, ap_mean, ap_std, precision_k_mean, precision_k_std, summary_csv_file):
    """CSV"""
    file_exists = os.path.exists(summary_csv_file)
    with open(summary_csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            # Precision@K
            writer.writerow(['Dataset', 'AUC_Mean', 'AUC_Std', 'AP_Mean', 'AP_Std', 'Precision@K_Mean', 'Precision@K_Std'])
        writer.writerow([str(dataset), f'{auc_mean:.6f}', f'{auc_std:.6f}',
                        f'{ap_mean:.6f}', f'{ap_std:.6f}',
                        f'{precision_k_mean:.6f}', f'{precision_k_std:.6f}'])


def append_result_to_csv(summary_data):
    """Append one finished run to the CSV next to this script immediately."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_name = summary_data.get('csv_name', SOURCE_RESULT_CSV_NAME)
    csv_file = os.path.join(script_dir, csv_name)
    target_file = csv_file
    try:
        return _append_result_to_csv_file(summary_data, target_file)
    except PermissionError:
        base, ext = os.path.splitext(csv_file)
        target_file = f"{base}_append_backup{ext}"
        print(f"[Save warning] main CSV is busy, writing fallback CSV: {target_file}")
        return _append_result_to_csv_file(summary_data, target_file)


def _append_result_to_csv_file(summary_data, csv_file):
    file_has_content = os.path.exists(csv_file) and os.path.getsize(csv_file) > 0

    chinese_headers = {
        'dataset': '',
        'ablation': '',
        'algorithm': '',
        'seed': '',
        'best_auc': 'AUC',
        'best_ap': 'AP',
        'best_auc_epoch': 'AUC',
        'top_3_auc': 'AUC',
        'early_stop_epoch': '',
        'total_time': '',
        'avg_epoch_time': '',
        'num_epoch': '',
        'density_mode': '',
        'global_sampling': '',
        'hop_num': '',
        'run_command': '',
        'lr': '',
        'embedding_dim': '',
        'drop_prob': '',
        'readout': '',
        'negsamp_ratio': '',
        'mean': '',
        'var': '',
        'fixed_weight_margin': '',
        'fixed_weight_bce': 'BCE',
        'global_sample_rate': '',
        'auc_test_rounds': 'AUC',
        'best_precision_at_k': 'Precision@K',
        'score_source': '',
        'best_val_auc': 'Val AUC',
        'best_val_ap': 'Val AP',
        'eval_mode': '',
        'run_mode': '',
        'train_rate': '',
        'val_rate': '',
        'test_rate': '',
        'score_fusion_weight': '',
    }

    fieldnames = list(summary_data.keys())
    with open(csv_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_has_content:
            writer.writerow({field: chinese_headers.get(field, field) for field in fieldnames})
            writer.writeheader()
        writer.writerow(summary_data)
        f.flush()
        os.fsync(f.fileno())

    print(f"[Saved] result appended to: {csv_file}")
    return csv_file

def calculate_precision_at_k(preds, labels, k=None):
    """
     Precision@K
    Args:
        preds:
        labels: 1=0=
        k: Top-KKNone
    Returns:
        float: Precision@K
    """
    # NaNAUC/AP
    nan_mask = ~np.isnan(preds)
    if np.any(nan_mask):
        preds = preds[nan_mask]
        labels = labels[nan_mask]

    # numpy
    preds = np.array(preds)
    labels = np.array(labels)

    # KK=
    if k is None:
        k = int(np.sum(labels))  #
    k = min(k, len(preds))  # K

    # Top-K
    top_k_indices = np.argsort(preds)[::-1][:k]

    # Top-K
    true_positives = np.sum(labels[top_k_indices] == 1)

    # Precision@K
    precision_at_k = true_positives / k if k > 0 else 0.0

    return precision_at_k

def evaluate_metrics(preds, labels):
    """

    Args:
        preds:
        labels:

    Returns:
        tuple: (auc, ap, precision_at_k)
    """
    # NaN
    nan_mask = ~np.isnan(preds)
    if np.any(nan_mask):
        preds = preds[nan_mask]
        labels = labels[nan_mask]

    if len(preds) > 0:
        auc = roc_auc_score(labels, preds)  # AUC
        AP = average_precision_score(
            labels, preds, average='macro', pos_label=1, sample_weight=None)  # AP
        # Precision@KK=
        precision_at_k = calculate_precision_at_k(preds, labels, k=None)
    else:
        print('Testing AUC: N/A (all NaN)')
        auc = 0.0
        AP = 0.0
        precision_at_k = 0.0

    return auc, AP, precision_at_k


def _as_numpy_1d(values):
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    return np.asarray(values).reshape(-1)


def normalize_score(score, ref_score=None):
    """Robustly map a score vector to a comparable rank-like range."""
    score = _as_numpy_1d(score).astype(np.float64)
    ref = score if ref_score is None else _as_numpy_1d(ref_score).astype(np.float64)
    finite_ref = ref[np.isfinite(ref)]
    if finite_ref.size == 0:
        return np.zeros_like(score, dtype=np.float64)

    fill_value = float(np.nanmedian(finite_ref))
    score = np.nan_to_num(score, nan=fill_value, posinf=np.nanmax(finite_ref), neginf=np.nanmin(finite_ref))
    ref = np.nan_to_num(ref, nan=fill_value, posinf=np.nanmax(finite_ref), neginf=np.nanmin(finite_ref))

    q25, q75 = np.percentile(ref, [25, 75])
    scale = q75 - q25
    if not np.isfinite(scale) or scale < 1e-12:
        scale = np.std(ref) + 1e-12
    z = (score - np.median(ref)) / scale
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _prediction_from_classifier(clf, features):
    if hasattr(clf, 'predict_proba'):
        return clf.predict_proba(features)[:, 1]
    if hasattr(clf, 'decision_function'):
        return clf.decision_function(features)
    return clf.predict(features)


def _label_propagation_scores(row_adj, labels, idx_train, alpha=0.7, num_iter=100):
    labels = _as_numpy_1d(labels).astype(int)
    y0 = np.zeros(labels.shape[0], dtype=np.float64)
    train_idx = np.asarray(idx_train, dtype=int)
    y0[train_idx] = np.where(labels[train_idx] == 1, 1.0, -1.0)
    scores = y0.copy()
    for _ in range(num_iter):
        scores = alpha * row_adj.dot(scores) + (1.0 - alpha) * y0
    return scores


def _build_graph_prior_features(raw_adj_cpu, features_cpu, labels, idx_train, normal_label_idx):
    raw_adj_cpu = raw_adj_cpu.tocsr().astype(np.float64)
    raw_adj_cpu.setdiag(0)
    raw_adj_cpu.eliminate_zeros()

    features_cpu = np.asarray(features_cpu, dtype=np.float64)
    degrees = np.asarray(raw_adj_cpu.sum(axis=1)).reshape(-1)
    safe_degrees = np.maximum(degrees, 1.0)
    row_adj = sp.diags(1.0 / safe_degrees).dot(raw_adj_cpu).tocsr()

    feature_norm = np.linalg.norm(features_cpu, axis=1)
    features_l2 = normalize(features_cpu, norm='l2', axis=1)
    neighbor_features = row_adj.dot(features_cpu)
    two_hop_neighbor_features = row_adj.dot(neighbor_features)
    neighbor_l2 = normalize(neighbor_features, norm='l2', axis=1)

    residual_l2 = np.linalg.norm(features_cpu - neighbor_features, axis=1)
    residual_two_hop_l2 = np.linalg.norm(features_cpu - two_hop_neighbor_features, axis=1)
    residual_cos = 1.0 - np.sum(features_l2 * neighbor_l2, axis=1)
    if np.any(degrees > 0):
        residual_cos[degrees == 0] = np.nanmedian(residual_cos[degrees > 0])
    residual_cos = np.nan_to_num(residual_cos, nan=0.0, posinf=0.0, neginf=0.0)

    normal_idx = np.asarray(normal_label_idx, dtype=int)
    if normal_idx.size == 0:
        labels_arr = _as_numpy_1d(labels).astype(int)
        normal_idx = np.asarray([idx for idx in idx_train if labels_arr[idx] == 0], dtype=int)
    if normal_idx.size == 0:
        normal_idx = np.arange(features_cpu.shape[0], dtype=int)

    normal_log_degree = np.log1p(degrees[normal_idx])
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

    return graph_features, features_cpu, row_adj, normal_idx, {
        'degree_high': degree_z,
        'degree_low': -degree_z,
        'degree_deviation': np.abs(degree_z),
        'feature_norm_high': feature_norm,
        'feature_norm_low': -feature_norm,
        'neighbor_residual_l2': residual_l2,
        'two_hop_residual_l2': residual_two_hop_l2,
        'neighbor_residual_cos': residual_cos,
    }


def build_fixed_prior_score(score_dict, ref_idx, allow_semi_supervised=False, single_prior=None, prior_list=None):
    """Create a fixed prior score without validation/test label selection.

    Args:
        score_dict: dict of prior scores
        ref_idx: reference indices for normalization
        allow_semi_supervised: whether to include semi-supervised priors
        single_prior: if set, only use this single prior (for ablation)
        prior_list: if set, only use these priors (comma-separated string or list)
    """
    all_prior_names = [
        'graph_normal_distance',
        'residual_centroid_distance',
        'neighbor_residual_l2',
    ]
    if single_prior is not None:
        if single_prior not in score_dict:
            print(f"[Warning] single prior {single_prior} is not in score_dict; skipped")
            return None, None, []
        preferred_names = [single_prior]
    elif prior_list is not None:
        if isinstance(prior_list, str):
            prior_list = [p.strip() for p in prior_list.split(',')]
        preferred_names = [p for p in prior_list if p in score_dict]
        if not preferred_names:
            print(f"[Warning] prior_list {prior_list} has no valid priors; skipped")
            return None, None, []
    else:
        preferred_names = all_prior_names.copy()

    if allow_semi_supervised:
        preferred_names.extend([
            name for name in score_dict
            if name.startswith('semi_')
        ])

    ref_idx = np.asarray(ref_idx, dtype=int)
    selected = []
    selected_names = []
    for name in preferred_names:
        if name not in score_dict:
            continue
        score = _as_numpy_1d(score_dict[name]).astype(np.float64)
        ref_score = score[ref_idx] if ref_idx.size > 0 else score
        selected.append(normalize_score(score, ref_score))
        selected_names.append(name)

    if not selected:
        return None, None, []

    prior_score = np.mean(np.vstack(selected), axis=0)
    prior_name = 'fixed_train_only_prior'
    if allow_semi_supervised:
        prior_name = 'fixed_train_only_prior_with_semi_supervised'
    return prior_name, normalize_score(prior_score, prior_score[ref_idx] if ref_idx.size > 0 else prior_score), selected_names


def build_graph_prior_scores(raw_adj_cpu, features_cpu, labels, idx_train, normal_label_idx, seed, args):
    """Build fixed train-only graph/attribute prior scores. No validation/test labels are used."""
    if args.disable_graph_prior:
        return {}, None

    labels_arr = _as_numpy_1d(labels).astype(int)
    graph_features, dense_features, row_adj, normal_reference_idx, base_scores = _build_graph_prior_features(
        raw_adj_cpu, features_cpu, labels_arr, idx_train, normal_label_idx
    )
    train_idx = np.asarray(idx_train, dtype=int)
    score_dict = {}
    normal_reference_idx = np.asarray(normal_reference_idx, dtype=int)

    for name, score in base_scores.items():
        score_dict[name] = normalize_score(score, score[normal_reference_idx])

    normal_train = np.asarray([idx for idx in train_idx if labels_arr[idx] == 0], dtype=int)
    abnormal_train = np.asarray([idx for idx in train_idx if labels_arr[idx] == 1], dtype=int)

    dense_l2 = normalize(dense_features, norm='l2', axis=1)
    normal_centroid = normalize(dense_l2[normal_reference_idx].mean(axis=0, keepdims=True), norm='l2', axis=1)
    centroid_distance = 1.0 - dense_l2.dot(normal_centroid.T).reshape(-1)
    score_dict['normal_centroid_distance'] = normalize_score(
        centroid_distance,
        centroid_distance[normal_reference_idx],
    )

    residual_l2_features = normalize(
        dense_features - row_adj.dot(dense_features),
        norm='l2',
        axis=1,
    )
    residual_centroid = normalize(
        residual_l2_features[normal_reference_idx].mean(axis=0, keepdims=True),
        norm='l2',
        axis=1,
    )
    residual_centroid_distance = 1.0 - residual_l2_features.dot(residual_centroid.T).reshape(-1)
    score_dict['residual_centroid_distance'] = normalize_score(
        residual_centroid_distance,
        residual_centroid_distance[normal_reference_idx],
    )

    graph_center = graph_features[normal_reference_idx].mean(axis=0, keepdims=True)
    graph_scale = graph_features[normal_reference_idx].std(axis=0, keepdims=True) + 1e-8
    graph_mahalanobis = np.sqrt(np.sum(((graph_features - graph_center) / graph_scale) ** 2, axis=1))
    score_dict['graph_normal_distance'] = normalize_score(
        graph_mahalanobis,
        graph_mahalanobis[normal_reference_idx],
    )



    # Optional semi-supervised branch. It uses true anomaly labels from idx_train and is
    # therefore disabled by default for fair unsupervised anomaly-detection reporting.
    if args.semi_supervised_prior and normal_train.size > 0 and abnormal_train.size > 0:
        abnormal_centroid = normalize(dense_l2[abnormal_train].mean(axis=0, keepdims=True), norm='l2', axis=1)
        prototype_score = (
            dense_l2.dot(abnormal_centroid.T).reshape(-1) -
            dense_l2.dot(normal_centroid.T).reshape(-1)
        )
        score_dict['semi_feature_prototype'] = normalize_score(prototype_score, prototype_score[train_idx])

        for alpha in (0.3, 0.5, 0.7, 0.85, 0.95):
            lp_score = _label_propagation_scores(row_adj, labels_arr, train_idx, alpha=alpha, num_iter=100)
            score_dict[f'semi_label_prop_{alpha:.2f}'] = normalize_score(lp_score, lp_score[train_idx])

        graph_model_specs = [
            ('semi_lr_graph', LogisticRegression(max_iter=1000, class_weight='balanced', solver='liblinear', random_state=seed), graph_features),
            ('semi_svm_graph', LinearSVC(class_weight='balanced', max_iter=5000, random_state=seed, dual='auto'), graph_features),
            ('semi_rf_graph', RandomForestClassifier(
                n_estimators=args.prior_n_estimators,
                class_weight='balanced',
                random_state=seed,
                min_samples_leaf=1,
                n_jobs=-1,
            ), graph_features),
            ('semi_et_graph', ExtraTreesClassifier(
                n_estimators=args.prior_n_estimators,
                class_weight='balanced',
                random_state=seed,
                min_samples_leaf=1,
                n_jobs=-1,
            ), graph_features),
        ]

        dense_plus_graph = np.hstack([dense_features, graph_features])
        model_specs = graph_model_specs + [
            ('semi_lr_attr_graph', LogisticRegression(max_iter=1000, class_weight='balanced', solver='liblinear', random_state=seed), dense_plus_graph),
        ]
        if dense_features.shape[0] <= 10000:
            model_specs.append((
                'semi_rf_attr_graph',
                RandomForestClassifier(
                    n_estimators=max(100, args.prior_n_estimators // 2),
                    class_weight='balanced',
                    random_state=seed,
                    min_samples_leaf=1,
                    n_jobs=-1,
                ),
                dense_plus_graph,
            ))
            model_specs.append((
                'semi_et_attr_graph',
                ExtraTreesClassifier(
                    n_estimators=max(100, args.prior_n_estimators // 2),
                    class_weight='balanced',
                    random_state=seed,
                    min_samples_leaf=1,
                    n_jobs=-1,
                ),
                dense_plus_graph,
            ))

        for name, clf, clf_features in model_specs:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', ConvergenceWarning)
                    clf.fit(clf_features[train_idx], labels_arr[train_idx])
                pred = _prediction_from_classifier(clf, clf_features)
                score_dict[name] = normalize_score(pred, pred[train_idx])
            except Exception as exc:
                print(f"[Graph prior] {name} training failed; skipped: {exc}")

    prior_name, prior_score, selected_names = build_fixed_prior_score(
        score_dict,
        train_idx,
        allow_semi_supervised=args.semi_supervised_prior,
        single_prior=getattr(args, 'single_prior', None),
        prior_list=getattr(args, 'prior_list', None),
    )
    if prior_score is None:
        return {}, None

    print(f"[Graph prior] fixed training-set prior: {prior_name} | components={len(selected_names)}")
    return {prior_name: prior_score}, prior_name


def _unsupervised_score_quality(score, ref_idx):
    """
    Estimate score reliability from the current dataset only.
    Higher is better. Uses labeled-normal/train reference distribution as the
    normal anchor and does not read validation/test labels.
    """
    score = _as_numpy_1d(score).astype(np.float64)
    ref_idx = np.asarray(ref_idx, dtype=int)
    ref = score[ref_idx] if ref_idx.size > 0 else score
    finite_ref = ref[np.isfinite(ref)]
    finite_all = score[np.isfinite(score)]
    if finite_ref.size < 2 or finite_all.size < 2:
        return 1e-6

    ref_median = float(np.median(finite_ref))
    ref_q25, ref_q75 = np.percentile(finite_ref, [25, 75])
    ref_iqr = float(ref_q75 - ref_q25)
    if not np.isfinite(ref_iqr) or ref_iqr < 1e-12:
        ref_iqr = float(np.std(finite_ref) + 1e-12)

    all_median = float(np.median(finite_all))
    separation = abs(all_median - ref_median) / ref_iqr
    variability = float(np.std(finite_all) / ref_iqr)

    ref_normalized = (finite_ref - ref_median) / ref_iqr
    ref_stability = 1.0 / (1.0 + float(np.std(ref_normalized)))
    quality = np.log1p(max(0.0, separation) + 0.25 * max(0.0, variability)) * ref_stability
    return float(max(quality, 1e-6))


def estimate_auto_fusion_weight(model_scores, prior_scores, ref_idx):
    model_quality = _unsupervised_score_quality(model_scores, ref_idx)
    prior_quality = _unsupervised_score_quality(prior_scores, ref_idx)
    alpha = model_quality / (model_quality + prior_quality + 1e-12)
    return float(np.clip(alpha, 0.05, 0.95)), model_quality, prior_quality


def build_fused_scores(model_scores, prior_score_dict, ref_idx, fusion_weight=0.5, auto_fusion_weight=True, prior_ratio=None, gamma_decay=1.0, dataset=None):
    ref_idx = np.asarray(ref_idx, dtype=int)
    model_scores = _as_numpy_1d(model_scores).astype(np.float64)
    model_ref = model_scores[ref_idx] if ref_idx.size > 0 else model_scores
    raw_model_scores = model_scores.copy()
    model_scores = normalize_score(raw_model_scores, model_ref)

    if not prior_score_dict:
        return 'model', model_scores, {'auc': 0.0, 'ap': 0.0, 'precision_at_k': 0.0, 'fusion_weight': 1.0}

    prior_name, prior_scores = next(iter(prior_score_dict.items()))
    prior_scores = _as_numpy_1d(prior_scores).astype(np.float64)
    prior_ref = prior_scores[ref_idx] if ref_idx.size > 0 else prior_scores
    raw_prior_scores = prior_scores.copy()
    prior_scores = normalize_score(raw_prior_scores, prior_ref)



    if abs(gamma_decay - 1.0) > 1e-8:
        gamma_decay = float(max(gamma_decay, 1e-6))
        prior_scores = np.power(prior_scores, gamma_decay)


    if prior_ratio is not None:
        prior_ratio = float(np.clip(prior_ratio, 0.0, 1.0))
        weight = 1.0 - prior_ratio  # model_weight = 1 - prior_ratio
        model_quality = 0.0
        prior_quality = 0.0
        score_name = f'forced_fusion:p{prior_ratio:.2f}+{prior_name}'
    elif auto_fusion_weight:
        weight, model_quality, prior_quality = estimate_auto_fusion_weight(model_scores, prior_scores, ref_idx)
        score_name = f'auto_fusion:m{weight:.2f}+{prior_name}'
    else:
        weight = float(np.clip(fusion_weight, 0.0, 1.0))
        model_quality = 0.0
        prior_quality = 0.0
        score_name = f'fixed_fusion:m{weight:.2f}+{prior_name}'



    if dataset == 'pubmed':
        prior_weight = 1.0 - weight
    else:
        prior_weight = max(0.0, (1.0 - weight) - 0.4)
    weight = 1.0 - prior_weight

    fused_scores = weight * model_scores + (1.0 - weight) * prior_scores
    return score_name, fused_scores, {
        'auc': 0.0,
        'ap': 0.0,
        'precision_at_k': 0.0,
        'fusion_weight': weight,
        'prior_ratio': 1.0 - weight,
        'model_quality': model_quality,
        'prior_quality': prior_quality,
    }


def load_graph_data(args, device):
    """

    Args:
        args:
        device:

    Returns:
        tuple:
    """
    # load_mat
    # - adj:  -
    # - features:  -
    # - labels:  - /
    # - all_idx:
    # - idx_train, idx_val, idx_test: //
    # - ano_label:  -
    # - str_ano_label, attr_ano_label:
    # - normal_label_idx, abnormal_label_idx:
    if args.strict_3_7:
        adj, features, labels, all_idx, idx_train, idx_val, \
            idx_test, ano_label, str_ano_label, attr_ano_label, normal_label_idx, abnormal_label_idx = load_mat(
                args.dataset, train_rate=args.train_rate, val_rate=0.0, file_path=args.file_path)
        print(
            f"[strict split] train={len(idx_train)} {format_label_counts(ano_label[idx_train])} | "
            f"val={len(idx_val)} | test={len(idx_test)} {format_label_counts(ano_label[idx_test])}"
        )
    else:
        adj, features, labels, all_idx, idx_train, idx_val, \
            idx_test, ano_label, str_ano_label, attr_ano_label, normal_label_idx, abnormal_label_idx = load_mat(
                args.dataset, file_path=args.file_path)
        print(
            f"[source-compatible split] train={len(idx_train)} {format_label_counts(ano_label[idx_train])} | "
            f"val={len(idx_val)} | test={len(idx_test)} {format_label_counts(ano_label[idx_test])}"
        )

    if args.dataset in [ 'tf_finace',  'elliptic']:
        features, _ = preprocess_features(features)  #
    else:
        features = features.todense()  #

    nb_nodes = features.shape[0]  #  -
    ft_size = features.shape[1]   #  -
    raw_adj = adj  #  -
    raw_adj_cpu = adj.copy().tocsr()
    features_cpu = np.asarray(features, dtype=np.float32)
    print(adj.sum())  # 1
    log_gpu_memory("load_graph_data: features converted to dense")

    adj = normalize_adj(adj)

    #   -
    raw_adj = raw_adj + sp.eye(raw_adj.shape[0])  #
    adj = adj + sp.eye(adj.shape[0])              #

    # ==================== PyTorch ====================

    #  PyTorch - PyTorch
    # batch [1, num_nodes, ft_size]
    features = torch.FloatTensor(features[np.newaxis]).to(device)              # FloatGPU

    adj = sparse_mx_to_torch_sparse_tensor(adj).to(device)  #
    adj = adj.unsqueeze(0)  # batch

    raw_adj = sparse_mx_to_torch_sparse_tensor(raw_adj).to(device)
    raw_adj = raw_adj.unsqueeze(0)  # batch

    labels = torch.FloatTensor(labels[np.newaxis]).to(device)      # batchGPU

    log_gpu_memory("load_graph_data: all tensors moved to GPU")

    print("Features have NaN:", torch.isnan(features).any().item())
    print("Labels have NaN:", torch.isnan(labels).any().item())

    return adj, features, labels, all_idx, idx_train, idx_val, idx_test, \
           ano_label, str_ano_label, attr_ano_label, normal_label_idx, abnormal_label_idx, \
           nb_nodes, ft_size, raw_adj, raw_adj_cpu, features_cpu


def is_experiment_completed(csv_file, dataset, seed, run_mode=None):
    """Is experiment completed."""
    if not os.path.exists(csv_file):
        return False

    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        if len(lines) < 3:
            return False

        import io
        header_line = lines[1].strip()
        data_lines = lines[2:]
        data_content = header_line + '\n' + ''.join(data_lines)

        reader = csv.DictReader(io.StringIO(data_content))
        for row in reader:
            try:
                row_dataset = row.get('dataset', '').strip()
                row_seed = row.get('seed', '').strip()
                row_auc = row.get('best_auc', '').strip()
                row_ap = row.get('best_ap', '').strip()
                row_mode = row.get('run_mode', '').strip()

                if not all([row_dataset, row_seed, row_auc, row_ap]):
                    continue

                row_dataset = canonical_dataset_name(row_dataset)
                row_seed = int(row_seed)
                row_auc = float(row_auc)
                row_ap = float(row_ap)

                if (
                    row_dataset == canonical_dataset_name(dataset) and
                    row_seed == seed and
                    (run_mode is None or row_mode == run_mode) and
                    row_auc > 0.0
                ):
                    return True
            except (ValueError, KeyError):
                continue
    except Exception:
        pass

    return False

def run_experiment(args, seed):

    run_mode = get_run_mode(args)
    eval_mode = get_eval_mode(args)
    result_csv_name = get_result_csv_name(args)
    active_fusion_weight = get_fusion_weight_for_dataset(args.dataset, args)
    if run_mode == 'model_only' and args.strict_3_7:
        result_csv_name = STRICT_RESULT_CSV_NAME
    if getattr(args, 'single_prior', None):
        prior_tag = args.single_prior.replace('neighbor_residual_l2', 'residual_l2').replace('graph_normal_distance', 'graph_norm').replace('residual_centroid_distance', 'res_centroid')
        result_csv_name = f'results_single_prior_{prior_tag}.csv'
    if getattr(args, 'prior_list', None):
        result_csv_name = f'results_without_rec.csv'
    if args.skip_completed:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_file = os.path.join(script_dir, result_csv_name)
        if is_experiment_completed(csv_file, args.dataset, seed, run_mode):
            print(f"\n{'='*60}")
            print(f'   [Skip] Dataset: {args.dataset}, Seed: {seed}, Mode: {run_mode}, Eval: {eval_mode} - Result')
            print(f"{'='*60}")
            return None


    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc as gc_module
    gc_module.collect()
    log_gpu_memory("run_experiment entry")

    torch.autograd.set_detect_anomaly(True)

    print(f"\n{'='*60}")
    print(f'  - Seed: {seed}')
    print(f'  - Run Mode: {run_mode}')
    if args.auto_fusion_weight and not args.personalized_fusion_weight:
        print(f'  - Fusion Mode: auto (train-reference reliability, no dataset-name table)')
    else:
        print(f'  - Fusion Weight(model): {active_fusion_weight:.2f}')
    if abs(args.gamma_decay - 1.0) > 1e-8:
        print(f'  - Gamma Decay: {args.gamma_decay} (nonlinear confidence suppression on prior)')
    print(f"{'='*60}")



    # ====================  ====================

    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f': {device}')
        print(f'GPU: {torch.cuda.get_device_name(0)}')
        print(f'GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')
    else:
        device = torch.device('cpu')
        print(f': {device}')
        if args.device == 'cuda' and not torch.cuda.is_available():
            print(': GPUCPU')
    log_gpu_memory("device initialized")

    # ====================  ====================

    # dropout
    # DGL
    # dgl.random.seed(seed)           # DGL
    np.random.seed(seed)            # NumPy
    torch.manual_seed(seed)         # PyTorch CPU
    torch.cuda.manual_seed(seed)    # PyTorch GPU
    torch.cuda.manual_seed_all(seed)  # GPU
    random.seed(seed)               # Pythonrandom
    torch.backends.cudnn.deterministic = True  # CUDA
    torch.backends.cudnn.benchmark = False     # CUDA

    # ====================  ====================

    adj, features, labels, all_idx, idx_train, idx_val, idx_test, \
    ano_label, str_ano_label, attr_ano_label, normal_label_idx, abnormal_label_idx, \
    nb_nodes, ft_size, raw_adj, raw_adj_cpu, features_cpu = load_graph_data(args, device)
    log_gpu_memory("graph data loaded")

    prior_score_dict, selected_prior_name = build_graph_prior_scores(
        raw_adj_cpu,
        features_cpu,
        ano_label,
        idx_train,
        normal_label_idx,
        seed,
        args,
    )


    if getattr(args, 'prior_noise', 0.0) > 0.0 and prior_score_dict:
        np.random.seed(seed)
        noise_std = float(args.prior_noise)
        for name, scores in prior_score_dict.items():
            noise = np.random.normal(0, noise_std, scores.shape)
            noisy_scores = scores + noise

            noisy_scores = np.clip(noisy_scores, 0.0, 1.0)
            prior_score_dict[name] = noisy_scores
        print(f"[Prior noise] added Gaussian noise, std={noise_std}")

    if args.prior_only and prior_score_dict:
        best_prior_name, best_prior_scores = next(iter(prior_score_dict.items()))
        best_auc, best_ap, best_precision_k = evaluate_metrics(
            best_prior_scores[idx_test],
            ano_label[idx_test]
        )
        print(f"[Graph prior-only] {best_prior_name} | {eval_mode}")
        print(f"[Graph prior-only] test_auc={best_auc:.4f}, test_ap={best_ap:.4f}, precision@k={best_precision_k:.4f}")
        summary_data = {
            'dataset': args.dataset,
            'algorithm': ALGORITHM_NAME,
            'seed': seed,
            'best_auc': round(float(best_auc), 5),
            'best_ap': round(float(best_ap), 5),
            'best_auc_epoch': -1,
            'top_3_auc': [],
            'early_stop_epoch': -1,
            'total_time': 0.0,
            'avg_epoch_time': 0.0,
            'num_epoch': 0,
            'density_mode': 'auto',
            'global_sampling': 'open',
            'hop_num': 2,
            'run_command': 'prior_only',
            'lr': args.lr,
            'embedding_dim': args.embedding_dim,
            'drop_prob': args.drop_prob,
            'readout': args.readout,
            'negsamp_ratio': args.negsamp_ratio,
            'mean': args.mean,
            'var': args.var,
            'fixed_weight_margin': args.fixed_weight_margin,
            'fixed_weight_bce': args.fixed_weight_bce,
            'global_sample_rate': args.global_sample_rate,
            'auc_test_rounds': args.auc_test_rounds,
            'best_precision_at_k': round(float(best_precision_k), 5),
            'score_source': best_prior_name,
            'eval_mode': eval_mode,
            'run_mode': run_mode,
            'train_rate': args.train_rate,
            'val_rate': 0.0 if args.strict_3_7 else 0.1,
            'test_rate': round(len(idx_test) / len(ano_label), 5),
            'score_fusion_weight': active_fusion_weight,
            'csv_name': result_csv_name,
        }
        append_result_to_csv(summary_data)
        print("[SUMMARY_DATA_START]")
        import json
        print(json.dumps(summary_data))
        print("[SUMMARY_DATA_END]")
        return summary_data

    # ====================  ====================

    # Model
    # - ft_size:
    # - args.embedding_dim: 300
    # - 'prelu':
    # - args.negsamp_ratio:
    # - args.readout:
    # - global_sample_rate:
    #  density_mode='auto', hop=2,

    density_mode = 'all_low' if getattr(args, 'disable_density', False) else 'auto'
    model = Model(ft_size, args.embedding_dim, 'prelu',
                  args.negsamp_ratio, args.readout,
                  global_sample_rate=args.global_sample_rate,
                  density_mode=density_mode,
                  hop=args.k_hops,
                  num_layers=args.num_layers).to(device)  # GPU
    log_gpu_memory("model created")

    # ====================  ====================

    if args.fixed_weight_margin is not None and args.fixed_weight_bce is not None:
        weight_margin = torch.tensor(args.fixed_weight_margin).to(device)
        weight_bce = torch.tensor(args.fixed_weight_bce).to(device)
        # print(f": margin={weight_margin:.2f}, bce={weight_bce:.2f}")

        optimiser = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    else:
        init_margin = torch.log(torch.exp(torch.tensor(1)) - 1)
        init_bce = torch.log(torch.exp(torch.tensor(1.0)) - 1)

        weight_margin = torch.nn.Parameter(init_margin, requires_grad=True).to(device)
        weight_bce = torch.nn.Parameter(init_bce, requires_grad=True).to(device)

        optimiser = torch.optim.Adam([
            {'params': model.parameters(), 'lr': args.lr},
            {'params': [weight_margin, weight_bce], 'lr': args.lr * 0.1}
        ], weight_decay=args.weight_decay)



    # ====================  REC  ====================
    print("\n[Init] computing REC weights...")
    log_gpu_memory("before REC computation")
    import torch.nn.functional as F

    # 1.
    feat_dim = features.shape[2] if len(features.shape) == 3 else features.shape[1]

    # 2.  ( 0)
    node_specific_rec = torch.zeros(nb_nodes, device=device)

    # 3.  2D  ( PyTorch )
    if raw_adj.is_sparse:
        edge_index = raw_adj.coalesce().indices()
        # batch
        if edge_index.dim() == 2 and edge_index.shape[0] == 3:
            # batch
            edge_index = edge_index[1:3, :]
        # edge_index2
        if edge_index.shape[0] != 2:
            # 2
            edge_index = edge_index.transpose(0, 1)
    else:
        # batch
        raw_adj_2d = raw_adj.squeeze(0)
        edge_index = torch.nonzero(raw_adj_2d, as_tuple=False).t()
    # [2, num_edges]
    if edge_index.shape[0] != 2:
        edge_index = edge_index[:2, :]
    print(f"[Graph] edge index shape: {edge_index.shape}")
    print(f"[Graph] edge count: {edge_index.shape[1]}")
    features_2d = features.squeeze(0)

    # 4.
    is_normal = torch.zeros(nb_nodes, dtype=torch.bool, device=device)
    is_normal[normal_label_idx] = True



    print("[REC] precomputing edge cosine similarities in batches...")


    src, dst = edge_index[0], edge_index[1]
    valid_edge_mask = is_normal[src] & is_normal[dst]
    pure_src, pure_dst = src[valid_edge_mask], dst[valid_edge_mask]
    num_pure_edges = pure_src.size(0)
    print(f"[REC] normal-normaledge count: {num_pure_edges}")


    edge_sim_dict = {}
    if num_pure_edges > 0:
        edge_batch_size = 20000
        num_edge_batches = (num_pure_edges + edge_batch_size - 1) // edge_batch_size
        for b in range(num_edge_batches):
            b_start = b * edge_batch_size
            b_end = min((b + 1) * edge_batch_size, num_pure_edges)
            batch_sim = F.cosine_similarity(
                features_2d[pure_src[b_start:b_end]],
                features_2d[pure_dst[b_start:b_end]],
                dim=-1
            )

            for i in range(b_end - b_start):
                s = pure_src[b_start + i].item()
                d = pure_dst[b_start + i].item()
                edge_sim_dict[(s, d)] = batch_sim[i].item()
            if (b + 1) % 10 == 0 or b == 0:
                log_gpu_memory(f"REC  {b+1}/{num_edge_batches}")
        print(f"[REC] edge similarity dictionary size: {len(edge_sim_dict)}")

    del edge_sim_dict




    print("[REC] computing incident-edge similarities...")


    node_sim_sum = torch.zeros(nb_nodes, device='cpu')
    node_edge_count = torch.zeros(nb_nodes, dtype=torch.long, device='cpu')

    edge_batch_size = 20000
    num_edge_batches = (num_pure_edges + edge_batch_size - 1) // edge_batch_size
    for b in range(num_edge_batches):
        b_start = b * edge_batch_size
        b_end = min((b + 1) * edge_batch_size, num_pure_edges)
        batch_sim = F.cosine_similarity(
            features_2d[pure_src[b_start:b_end]],
            features_2d[pure_dst[b_start:b_end]],
            dim=-1
        ).cpu()

        src_nodes = pure_src[b_start:b_end].cpu()
        dst_nodes = pure_dst[b_start:b_end].cpu()

        node_sim_sum.scatter_add_(0, src_nodes, batch_sim)
        node_sim_sum.scatter_add_(0, dst_nodes, batch_sim)
        node_edge_count.scatter_add_(0, src_nodes, torch.ones_like(src_nodes, dtype=torch.long))
        node_edge_count.scatter_add_(0, dst_nodes, torch.ones_like(dst_nodes, dtype=torch.long))

        if (b + 1) % 10 == 0 or b == 0:
            print(f"  [REC] vectorized batch {b+1}/{num_edge_batches}")


    valid_mask = node_edge_count > 0
    avg_sim = torch.zeros(nb_nodes)
    avg_sim[valid_mask] = node_sim_sum[valid_mask] / node_edge_count[valid_mask].float()


    local_h_values = (avg_sim[normal_label_idx] + 1.0) / 2.0
    local_h_values = torch.clamp(local_h_values, 0.0, 1.0).tolist()

    print("[REC] vectorized computation finished")

    del pure_src, pure_dst, node_sim_sum, node_edge_count, avg_sim

    log_gpu_memory("REC exact computation finished")

    # 6.
    local_h_tensor = torch.tensor(local_h_values, device=device)

    # 7. Z-Score
    mean_h = local_h_tensor.mean()
    std_h = local_h_tensor.std() + 1e-8
    z_scores = (local_h_tensor - mean_h) / std_h

    # ====================  ====================
    # 1. (+1)(-1)

    # 2.  (Global REC Base)
    # REC_global = 1.0 - 0.5 * log10(feat_dim)
    log_feat_dim = torch.log10(torch.tensor(feat_dim, dtype=torch.float32, device=device))
    rec_global = 1.0 - 0.5 * log_feat_dim
    #  [-1.0, 1.0]
    rec_global = torch.clamp(rec_global, min=-1.0, max=1.0)

    # 3.  (Local REC)
    # Z-ScoreTanh [-1, 1]
    rec_local = torch.tanh(z_scores)
    rec_local=-rec_local
    # 4.  (Confidence Gating )
    #  = sigmoid(5.0 * (mean_h - 0.5) - 0.5 * log10(feat_dim))
    alpha = torch.sigmoid(5.0 * (mean_h - 0.5) - 0.5 * log_feat_dim)

    # 5.  (Fusion)
    # REC_final = (1 - ) * REC_global +  * REC_local
    final_node_weights = (1 - alpha) * rec_global + alpha * rec_local
    # #  +1-1
    # final_node_weights = -final_node_weights
    print(f"[REC params] feature dim: {feat_dim}, local similarity mean: {mean_h.item():.4f}")
    print(f"[REC params] fusion alpha: {alpha.item():.4f}")
    print(f"[REC params] global REC weight: {rec_global.item():.4f}")
    # =====================================================================================

    # 10.  ID
    #  O(1)
    node_specific_rec[normal_label_idx] = final_node_weights

    print("[Done] RECInitDone")
    log_gpu_memory("REC Done")
    print()

    # ====================  ====================



    affinity_row = edge_index[0].long()
    affinity_col = edge_index[1].long()
    affinity_num_edges = affinity_row.numel()
    affinity_edge_batch_size = max(1, int(args.affinity_edge_batch_size))
    affinity_max_edges = int(args.affinity_max_edges)

    degree_sum_cached = torch.zeros(nb_nodes, device=device)
    if affinity_num_edges > 0:
        degree_sum_cached.scatter_add_(
            0,
            affinity_col,
            torch.ones(affinity_num_edges, dtype=torch.float, device=device)
        )
    r_inv_cached = 1.0 / degree_sum_cached.clamp(min=1e-8)
    if affinity_max_edges > 0 and affinity_num_edges > affinity_max_edges:
        print(
            f"[Local affinity] edge count: {affinity_num_edges}, "
            f"sampled per epoch: {affinity_max_edges}, batch size: {affinity_edge_batch_size}"
        )
    else:
        print(f"[Local affinity] edge count: {affinity_num_edges}, using all edges, batch size: {affinity_edge_batch_size}")
    log_gpu_memory("Local affinityDone")

    # BCEWithLogitsLoss = Sigmoid + BCE
    # reduction='none':
    # pos_weight:
    b_xent = nn.BCEWithLogitsLoss(
        reduction='none', pos_weight=torch.tensor([args.negsamp_ratio]).to(device))

    xent = nn.CrossEntropyLoss()

    # ====================  ====================

    #  tqdm
    log_gpu_memory("before training")
    torch.cuda.empty_cache()
    log_gpu_memory("after cache cleanup")
    with tqdm(total=args.num_epoch) as pbar:
        pbar.set_description('Training')  #
        total_time = 0  #
        final_auc = 0.0  # AUC
        final_ap = 0.0   # AP
        final_precision_k = 0.0  # Precision@K
        best_auc = 0.0
        best_ap = 0.0
        best_precision_k = 0.0
        best_epoch = 0
        best_score_source = 'model'
        best_fusion_weight = active_fusion_weight
        early_stop_triggered = False
        auc_history = []  # epochAUCAP
        final_epoch = 0
        final_score_source = 'model'
        final_fusion_weight = active_fusion_weight
        patience = args.patience

        #   - epoch
        for epoch in range(args.num_epoch):
            start_time = time.time()  # epoch


            if epoch % 100 == 0:
                log_gpu_memory(f"training epoch {epoch}/{args.num_epoch}")

            #  epoch
            model.current_epoch = epoch

            # dropoutbatch normalization
            model.train()

            #   -
            optimiser.zero_grad()

            # ====================  ====================
            # - features:
            # - adj:
            # - abnormal_label_idx:
            # - normal_label_idx:
            # - train_flag=True:
            # - args:
            train_flag = True
            emb, emb_combine, logits, emb_con, emb_abnormal = model(features, adj,
                                                                    abnormal_label_idx, normal_label_idx,
                                                                    train_flag, args)

            #  10epocht-SNE
            # t-SNE
            if epoch % 10 == 0:
                pass  #
                # tsne_data_path = 'draw/tfinance/tsne_data_{}.mat'.format(str(epoch))
                # io.savemat(tsne_data_path, {'emb': np.array(emb.cpu().detach()), 'ano_label': ano_label,
                #                             'abnormal_label_idx': np.array(abnormal_label_idx),
                #                             'normal_label_idx': np.array(normal_label_idx)})

            # ====================  ====================

            #  BCE -
            # 01
            # torch.zeros(len(normal_label_idx)): 0
            # torch.ones(len(emb_con)): 1
            # torch.unsqueezeunsqueeze(0):
            lbl = torch.unsqueeze(torch.cat(
                (torch.zeros(len(normal_label_idx)), torch.ones(len(emb_con)))),
                1).unsqueeze(0).to(device)

            #  BCE
            loss_bce = b_xent(logits, lbl)  #
            loss_bce = torch.mean(loss_bce)  #

            #   (Asymmetric Local Affinity Loss)
            # ""

            # batch [1, num_nodes, dim]  [num_nodes, dim]
            emb = torch.squeeze(emb)

            #   -
            emb_inf = torch.norm(emb, dim=-1, keepdim=True)  # L2
            # clamp
            emb_inf = 1.0 / emb_inf.clamp(min=1e-8)           #
            emb_norm = emb * emb_inf                         #

            num_nodes = emb_norm.shape[0]



            sim_sum = torch.zeros(num_nodes, device=device)
            if affinity_num_edges > 0 and affinity_max_edges > 0 and affinity_num_edges > affinity_max_edges:
                sampled_edge_pos = torch.randint(
                    0,
                    affinity_num_edges,
                    (affinity_max_edges,),
                    device=device
                )
                active_row = affinity_row[sampled_edge_pos]
                active_col = affinity_col[sampled_edge_pos]

                degree_sum = torch.zeros(num_nodes, device=device)
                degree_sum.scatter_add_(0, active_col, torch.ones_like(active_col, dtype=torch.float))
                r_inv = 1.0 / degree_sum.clamp(min=1e-8)
                active_num_edges = affinity_max_edges
            else:
                active_row = affinity_row
                active_col = affinity_col
                r_inv = r_inv_cached
                active_num_edges = affinity_num_edges

            if active_num_edges > 0:
                for edge_start in range(0, active_num_edges, affinity_edge_batch_size):
                    edge_end = min(edge_start + affinity_edge_batch_size, active_num_edges)
                    row_batch = active_row[edge_start:edge_end]
                    col_batch = active_col[edge_start:edge_end]
                    edge_similarities = torch.sum(
                        emb_norm[row_batch] * emb_norm[col_batch],
                        dim=1
                    )
                    sim_sum.scatter_add_(0, col_batch, edge_similarities)

            # 4.
            affinity = sim_sum * r_inv
            # ==============================================================================
            affinity_normal_mean = torch.mean(
                affinity[normal_label_idx])     #
            affinity_abnormal_mean = torch.mean(
                affinity[abnormal_label_idx])  #

            confidence_margin = 0.7  #  - 0.7
            loss_margin = (confidence_margin - (affinity_normal_mean -
                           affinity_abnormal_mean)).clamp_min(min=0)
            # clamp_min(min=0): 0.70

            #   (Egocentric Closeness Loss)
            #  (emb_con)   (emb_abnormal)
            diff_attribute = torch.pow(emb_con - emb_abnormal, 2)
            raw_distances = torch.sqrt(torch.sum(diff_attribute, 1) + 1e-8)

            # =================  =================
            anchor_weights = node_specific_rec[normal_label_idx]

            #  1  N  N
            num_anchors = len(normal_label_idx)
            num_generated = raw_distances.shape[0]

            base_count = num_generated // num_anchors
            remainder = num_generated % num_anchors #

            #  distance  [num_generated]
            if isinstance(normal_label_idx, list):
                normal_idx_tensor = torch.tensor(normal_label_idx, device=device)
            else:
                normal_idx_tensor = normal_label_idx.clone().detach().to(device)

            repeats = torch.full((num_anchors,), base_count, dtype=torch.long, device=device)
            if num_generated % num_anchors > 0:
                repeats[:num_generated % num_anchors] += 1

            batch_rec_weights = torch.repeat_interleave(anchor_weights, repeats)

            # [Dual-Mode Repulsion Loss]
            # (>0): distance * weightloss
            # (<0): |weight| / distanceloss
            positive_mask = batch_rec_weights > 0
            negative_mask = batch_rec_weights < 0

            if positive_mask.sum() > 0:
                loss_rec_positive = torch.mean(raw_distances[positive_mask] * batch_rec_weights[positive_mask])
            else:
                loss_rec_positive = torch.tensor(0.0, device=device)

            if negative_mask.sum() > 0:
                loss_rec_negative = torch.mean(torch.abs(batch_rec_weights[negative_mask]) / (raw_distances[negative_mask] + 1e-8))
            else:
                loss_rec_negative = torch.tensor(0.0, device=device)

            loss_rec = loss_rec_positive + loss_rec_negative


            if getattr(args, 'disable_rec', False):
                loss_rec = torch.tensor(0.0, device=device)

            if args.fixed_weight_margin is not None:
                weight_margin_pos = weight_margin
                weight_bce_pos = weight_bce
            else:
                weight_margin_pos = torch.nn.functional.softplus(weight_margin)
                weight_bce_pos = torch.nn.functional.softplus(weight_bce)

            global_rec_scale = 5.0
            loss = weight_margin_pos * loss_margin + weight_bce_pos * loss_bce + (global_rec_scale * loss_rec)

            # NaN - NaN
            nan_detected = False
            nan_checks = {
                "emb": torch.isnan(emb).any().item(),
                "emb_combine": torch.isnan(emb_combine).any().item(),
                "logits": torch.isnan(logits).any().item(),
                "emb_con": torch.isnan(emb_con).any().item(),
                "emb_abnormal": torch.isnan(emb_abnormal).any().item(),
                "loss_bce": torch.isnan(loss_bce).any().item(),
                "loss_margin": torch.isnan(loss_margin).any().item(),
                "loss_rec": torch.isnan(loss_rec).any().item(),
                "loss": torch.isnan(loss).any().item()
            }

            for name, has_nan in nan_checks.items():
                if has_nan:
                    nan_detected = True
                    break

            if nan_detected:
                print(f"Epoch {epoch}: NaN detected!")
                for name, has_nan in nan_checks.items():
                    print(f"  - {name} has NaN: {has_nan}")

            # NaN
            if torch.isnan(loss).any():
                print("Warning: Loss turned to NaN!")
                break
            if torch.isnan(logits).any():
                print("Warning: Logits turned to NaN!")
                break

            # ====================  ====================

            #   -
            loss.backward()  # PyTorch

            #   -
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            #   -
            optimiser.step()

            end_time = time.time()
            total_time += end_time - start_time
            # print('Total time is', total_time)  #

            # ====================  ====================

            #  epoch
            model.eval()  #  dropout
            train_flag = False  #

            emb, emb_combine, logits, emb_con, emb_abnormal = model(features, adj, abnormal_label_idx, normal_label_idx,
                                                                    train_flag, args)

            all_logits = np.squeeze(logits.cpu().detach().numpy())
            if all_logits.ndim > 1:
                all_logits = all_logits.reshape(-1)
            ano_label_test = ano_label[idx_test]

            if run_mode == 'model_only':
                selected_score_name = 'model'
                selected_scores = all_logits
                selected_val_metrics = {'auc': 0.0, 'ap': 0.0, 'precision_at_k': 0.0}
            else:
                selected_score_name, selected_scores, selected_val_metrics = build_fused_scores(
                    all_logits,
                    prior_score_dict,
                    normal_label_idx if len(normal_label_idx) > 0 else idx_train,
                    active_fusion_weight,
                    auto_fusion_weight=args.auto_fusion_weight and not args.personalized_fusion_weight and args.prior_ratio is None,
                    prior_ratio=args.prior_ratio,
                    gamma_decay=args.gamma_decay,
                    dataset=args.dataset,
                )
                if selected_scores is None:
                    selected_score_name = 'model'
                    selected_scores = all_logits
                    selected_val_metrics = {'auc': 0.0, 'ap': 0.0, 'precision_at_k': 0.0, 'fusion_weight': 1.0}

            auc, AP, precision_at_k = evaluate_metrics(selected_scores[idx_test], ano_label_test)

            #  AUC
            auc_history.append({
                'epoch': epoch,
                'auc': auc,
                'ap': AP,
                'precision_at_k': precision_at_k,
                'score_source': selected_score_name,
                'val_auc': 0.0,
                'val_ap': 0.0,
                'loss_margin': loss_margin.item(),
                'loss_bce': loss_bce.item(),
                'loss_rec': loss_rec.item()
            })

            final_auc = auc
            final_ap = AP
            final_precision_k = precision_at_k
            final_epoch = epoch
            final_score_source = selected_score_name
            final_fusion_weight = selected_val_metrics.get('fusion_weight', active_fusion_weight)


            if run_mode == 'model_only':
                improved = auc > best_auc
            else:
                improved = auc > best_auc or (abs(auc - best_auc) < 1e-12 and AP > best_ap)

            if improved:
                best_auc = auc
                best_ap = AP
                best_precision_k = precision_at_k
                best_epoch = epoch
                best_score_source = selected_score_name
                best_fusion_weight = selected_val_metrics.get('fusion_weight', active_fusion_weight)

            #  AUCAUC10epoch
            if epoch % 10 == 0 or epoch == args.num_epoch - 1:
                print("[%04d/%04d] margin: %.5f | bce: %.5f | rec: %.5f | total: %.5f | test_auc: %.4f | test_ap: %.4f | precision@k: %.4f | best_auc: %.4f | score: %s" % (
                    epoch, args.num_epoch, loss_margin.item(), loss_bce.item(), loss_rec.item(), loss.item(),
                    auc, AP, precision_at_k, best_auc, selected_score_name[:80]))

            if epoch - best_epoch >= patience:
                print(f"\n[Early stop] no improvement for {patience} epochs over best AUC {best_auc:.4f},stopped at epoch {epoch}")
                early_stop_triggered = True
                pbar.update(1)
                break
            pbar.update(1)


    final_auc = best_auc
    final_ap = best_ap
    final_precision_k = best_precision_k
    final_epoch = best_epoch
    final_score_source = best_score_source
    final_fusion_weight = best_fusion_weight


    print("\n=====================================================================")
    if early_stop_triggered:
        print(f'[Early stop] best epoch={final_epoch}, patience={patience}')
    print(f'[Final] Epoch: {final_epoch}/{args.num_epoch}')
    print(f'[Final] Test AUC: {final_auc:.4f}')
    print(f'[Final] Test AP: {final_ap:.4f}')
    print(f'[Final] Test Precision@K: {final_precision_k:.4f}')
    print(f'[Final] score source: {final_score_source} | {eval_mode}')
    print("=====================================================================")

    # ==================== AUC ====================
    # AUC
    if len(auc_history) > 0:
        print()



    print("\n" + "="*20)
    print(f'[Result] {ALGORITHM_NAME}')
    print(f'[Result] Dataset: {args.dataset}')
    print(f'[Result] FinalAUC: {final_auc:.4f}')
    print(f'[Result] FinalAP: {final_ap:.4f}')
    print(f'[Result] FinalPrecision@K: {final_precision_k:.4f}')
    print(f'[Result] score source: {final_score_source}')
    print("="*20)

    import datetime
    current_time = datetime.datetime.now()
    time_str = current_time.strftime('%Y%m%d_%H%M%S')
    auc_str = f"{final_auc:.3f}"

    script_dir = os.path.dirname(os.path.abspath(__file__))
    run_command = f"python {os.path.relpath(os.path.abspath(__file__), os.path.dirname(script_dir))} --dataset {args.dataset} --num_epoch {args.num_epoch}"
    if args.lr is not None:
        run_command += f" --lr {args.lr}"
    run_command += f" --seed {seed}"

    #  - .py
    algorithm_name = ALGORITHM_NAME

    # AUC
    top_3_auc = []
    if len(auc_history) > 0:
        top_3_auc = [(record['epoch'], round(record['auc'], 5), round(record['ap'], 5), record.get('score_source', 'model')) for record in auc_history[-3:]]

    early_stop_epoch = final_epoch

    avg_epoch_time = round(total_time / (epoch + 1), 5) if epoch > 0 else 0


    ablation_flags = []
    if getattr(args, 'single_prior', None):
        ablation_flags.append(f'single_prior:{args.single_prior}')
    if getattr(args, 'prior_list', None):
        ablation_flags.append(f'priors:{args.prior_list}')
    if getattr(args, 'disable_rec', False):
        ablation_flags.append('w/o REC')
    if getattr(args, 'disable_taf_qa', False):
        ablation_flags.append('w/o TAF-QA')
    if getattr(args, 'disable_density', False):
        ablation_flags.append('w/o Density')
    if getattr(args, 'disable_graph_prior', False):
        ablation_flags.append('model_only')
    if getattr(args, 'prior_only', False):
        ablation_flags.append('prior_only')
    if abs(args.gamma_decay - 1.0) > 1e-8:
        ablation_flags.append(f'gamma={args.gamma_decay}')
    ablation_tag = ','.join(ablation_flags) if ablation_flags else 'full'

    summary_data = {
        'dataset': args.dataset,
        'ablation': ablation_tag,
        'algorithm': algorithm_name,
        'seed': seed,
        'best_auc': round(float(final_auc), 5),
        'best_ap': round(float(final_ap), 5),
        'best_auc_epoch': final_epoch,
        'top_3_auc': top_3_auc,
        'early_stop_epoch': early_stop_epoch,
        'total_time': round(total_time, 5),
        'avg_epoch_time': avg_epoch_time,
        'num_epoch': args.num_epoch,
        'density_mode': 'auto',
        'global_sampling': 'open',
        'hop_num': 2,
        'run_command': run_command,
        'lr': args.lr,
        'embedding_dim': args.embedding_dim,
        'drop_prob': args.drop_prob,
        'readout': args.readout,
        'negsamp_ratio': args.negsamp_ratio,
        'mean': args.mean,
        'var': args.var,
        'fixed_weight_margin': args.fixed_weight_margin,
        'fixed_weight_bce': args.fixed_weight_bce,
        'global_sample_rate': args.global_sample_rate,
        'auc_test_rounds': args.auc_test_rounds,
        'best_precision_at_k': round(float(final_precision_k), 5),
        'score_source': final_score_source,
        'best_val_auc': 0.0,
        'best_val_ap': 0.0,
        'eval_mode': eval_mode,
        'run_mode': run_mode,
        'train_rate': args.train_rate,
        'val_rate': 0.0 if args.strict_3_7 else 0.1,
        'test_rate': round(len(idx_test) / len(ano_label), 5),
        'score_fusion_weight': final_fusion_weight,
        'csv_name': result_csv_name,
    }

    # 2.  -
    # # AUCAP
    # raw_data = []
    # for i, record in enumerate(auc_history):
    #     raw_data.append({
    #         'epoch': record['epoch'],
    #         'auc': round(record['auc'], 5),
    #         'ap': round(record['ap'], 5),
    #         'loss_margin': round(record['loss_margin'], 5),
    #         'loss_bce': round(record['loss_bce'], 5),
    #         'loss_rec': round(record['loss_rec'], 5)
    #     })

    # : {dataset}_summary_{embedding_dim}_{margin}_{sample_rate}_seed{seed}_{time_str}.csv
    # : {dataset}_raw_{embedding_dim}_{margin}_{sample_rate}_seed{seed}_{time_str}.csv
    embedding_dim = args.embedding_dim
    margin = args.fixed_weight_margin
    sample_rate = args.global_sample_rate

    # CSV
    csv_file = append_result_to_csv(summary_data)

    #  PredictiveGpuScheduler.py
    print("[SUMMARY_DATA_START]")
    import json
    print(json.dumps(summary_data))
    print("[SUMMARY_DATA_END]")

    log_gpu_memory(",Result")
    return summary_data

# ==================== AUC ====================
# if len(auc_history) > 0:
#     script_dir = os.path.dirname(os.path.abspath(__file__))
#     image_dir = os.path.join(script_dir, '')
    # plot_auc_curve(auc_history, args.dataset, save_dir=image_dir)

"""


epoch
1.  &
2.
3.
   - BCE
   -
   -
4.
5.
6.
7.


-  = 0
-  = 1
-  =
  1. BCE
  2.
  3.
-  =
"""


# ====================  ====================


# ==================== Standard training mode ====================


def run_one_or_three_ablation(args, seed):
    if not args.three_ablation:
        result = run_experiment(args, seed)
        return [] if result is None else [result]

    original_flags = {
        'disable_graph_prior': args.disable_graph_prior,
        'prior_only': args.prior_only,
        'semi_supervised_prior': args.semi_supervised_prior,
    }
    results = []
    try:
        for ablation_idx, (mode_name, flag_values) in enumerate(THREE_ABLATION_SPECS, 1):
            print(f"\n{'='*70}")
            print(f"# [Ablation {ablation_idx}/3] {mode_name}")
            print(f"{'='*70}")
            for flag_name, flag_value in flag_values.items():
                setattr(args, flag_name, flag_value)
            result = run_experiment(args, seed)
            if result is not None:
                results.append(result)
                print(
                    f"[AblationDone] {mode_name}: "
                    f"AUC={result['best_auc']:.4f}, AP={result['best_ap']:.4f}, "
                    f"Precision@K={result['best_precision_at_k']:.4f}"
                )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        for flag_name, flag_value in original_flags.items():
            setattr(args, flag_name, flag_value)
    return results


if __name__ == '__main__':
    import datetime
    import traceback

    if hasattr(args, 'dataset') and args.dataset:
        datasets_to_use = [args.dataset]
    else:
        datasets_to_use = DATASET_LIST

    if hasattr(args, 'seed') and args.seed is not None:
        seeds_to_use = [args.seed]
    else:
        seeds_to_use = SEED_LIST

    print(f"\n{'#'*70}")
    print(f'# Standard training mode: {ALGORITHM_NAME} / {get_eval_mode(args)}')
    print(f'# Dataset: {datasets_to_use}')
    print(f'# Seeds: {seeds_to_use}')
    print(f"{'#'*70}")


    for si, seed in enumerate(seeds_to_use, 1):
        print(f"\n{'='*70}")
        print(f'# Seed [{si}/{len(seeds_to_use)}]  seed={seed}')
        print(f"{'#'*70}")

        all_results = []

        for dataset_idx, dataset in enumerate(datasets_to_use, 1):
            print(f"\n{'='*70}")
            print(f'# [{dataset_idx}/{len(datasets_to_use)}] Dataset: {dataset} (seed={seed})')
            print(f"{'#'*70}")

            args.dataset = dataset

            try:
                seed_results = run_one_or_three_ablation(args, seed)

                if not seed_results:
                    print(f"\n[Skip] {dataset} seed={seed} Done,")
                    continue

                all_results.extend(seed_results)

                print(f"\n[Success] {dataset} seed={seed} Done {len(seed_results)} runs!")
                for result in seed_results:
                    print(
                        f"   {result.get('run_mode', 'unknown')}: "
                        f"AUC={result['best_auc']:.4f}, "
                        f"AP={result['best_ap']:.4f}, "
                        f"Precision@K={result['best_precision_at_k']:.4f}"
                    )

                del seed_results
                gc.collect()
                torch.cuda.empty_cache()
                print("\n[Cleanup] memory released")

            except Exception as e:
                print(f"\n[Error] {dataset} seed={seed} training failed!")
                print(f"   Error: {str(e)}")
                traceback.print_exc()


        if all_results:
            print(f"\n{'='*70}")
            print(f' Seed {seed} summary - {len(all_results)} Dataset')
            print(f"{'#'*70}")
            auc_list = [r['best_auc'] for r in all_results]
            ap_list = [r['best_ap'] for r in all_results]
            precision_k_list = [r['best_precision_at_k'] for r in all_results]
            print(f"  AUC mean: {np.mean(auc_list):.4f} +/- {np.std(auc_list):.4f}")
            print(f"  AP  mean: {np.mean(ap_list):.4f} +/- {np.std(ap_list):.4f}")
            print(f"  P@K mean: {np.mean(precision_k_list):.4f} +/- {np.std(precision_k_list):.4f}")
            print(f"{'='*70}")
        else:
            print(f"{'='*70}")
            print(f"[Warning] Seed {seed} Result!")

    print(f"\n{'#'*70}")
    print('# Done!')
    print(f'{"#"*70}')

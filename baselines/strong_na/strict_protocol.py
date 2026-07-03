"""Strict normal-only protocol utilities for strong classical baselines.

This module intentionally avoids the legacy baseline split helpers. The split is
the paper protocol used by the fixed DMC runner:

* train: 25% of normal nodes only
* test: remaining normal nodes plus all anomaly nodes
* validation labels: unused
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import scipy.io as sio
import scipy.sparse as sp
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import normalize


REPO_ROOT = Path(__file__).resolve().parents[2]


STRICT_PROTOCOL_NAME = "normal_25_train_all_anomaly_test"
LABEL_VISIBILITY = "normal_only_pseudo_negative"


@dataclass(frozen=True)
class StrictData:
    dataset: str
    seed: int
    adj: sp.csr_matrix
    features: sp.csr_matrix
    labels: np.ndarray
    idx_train: np.ndarray
    idx_test: np.ndarray
    normal_label_idx: np.ndarray


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def resolve_dataset_path(dataset: str, file_path: str | None = None) -> Path:
    if file_path:
        path = Path(file_path)
        if path.is_dir():
            for suffix in (".mat", ".npz", ".pkl"):
                candidate = path / f"{dataset}{suffix}"
                if candidate.exists():
                    return candidate
        if path.exists():
            return path

    dataset_path = Path(dataset)
    if dataset_path.exists():
        return dataset_path

    for suffix in (".mat", ".npz", ".pkl"):
        candidate = REPO_ROOT / "datasets" / f"{dataset}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Dataset file not found for {dataset} under {REPO_ROOT / 'datasets'}")


def _pick_first(mapping: dict, names: Iterable[str]):
    for name in names:
        if name in mapping and mapping[name] is not None:
            return mapping[name]
    return None


def _load_graph_file(path: Path) -> tuple[sp.csr_matrix, sp.csr_matrix, np.ndarray]:
    if path.suffix == ".mat":
        data = sio.loadmat(path)
        network = None
        if {"PAP", "PLP", "PTP"}.issubset(data):
            network = data["PAP"]
        elif {"net_APTPA", "net_APCPA", "net_APA"} & set(data):
            network = _pick_first(data, ["net_APA", "net_APCPA", "net_APTPA"])
        elif {"MAM", "MDM", "MYM"} & set(data):
            network = _pick_first(data, ["MAM", "MDM", "MYM"])
        if network is None:
            network = _pick_first(data, ["Network", "A", "adj", "homo", "adj_matrix", "network"])

        features = _pick_first(
            data,
            ["Attributes", "X", "features", "feature", "node_features", "feat", "attr"],
        )
        labels = _pick_first(data, ["Label", "gnd", "label", "labels", "node_labels", "y"])
    elif path.suffix == ".npz":
        data = dict(np.load(path, allow_pickle=True))
        network = _pick_first(data, ["adj", "adjacency", "A", "network", "homo"])
        if network is None:
            edge_index = _pick_first(data, ["edges", "edge_index"])
            if edge_index is not None:
                edge_index = np.asarray(edge_index)
                if edge_index.shape[0] == 2:
                    rows, cols = edge_index[0], edge_index[1]
                else:
                    rows, cols = edge_index[:, 0], edge_index[:, 1]
                n_nodes = int(max(rows.max(), cols.max())) + 1
                network = sp.coo_matrix(
                    (np.ones(len(rows), dtype=np.float32), (rows, cols)),
                    shape=(n_nodes, n_nodes),
                )
                network = network + network.T
        features = _pick_first(
            data,
            ["node_features", "features", "X", "attr", "attribute", "feat"],
        )
        labels = _pick_first(data, ["node_labels", "label", "labels", "Label", "gnd", "y"])
    else:
        raise ValueError(f"Unsupported dataset suffix: {path.suffix}")

    if network is None or features is None or labels is None:
        raise ValueError(f"{path} does not contain adjacency, features, and labels")

    adj = sp.csr_matrix(network).astype(np.float32)
    feat = sp.csr_matrix(features).astype(np.float32)
    labels = np.asarray(labels).reshape(-1).astype(int)
    if adj.shape[0] != feat.shape[0] or adj.shape[0] != labels.shape[0]:
        raise ValueError(
            f"{path} shape mismatch: adj={adj.shape}, features={feat.shape}, labels={labels.shape}"
        )
    return adj, feat, labels


def load_strict_data(dataset: str, seed: int, train_rate: float = 0.25, file_path: str | None = None) -> StrictData:
    """Load data with the strict normal-only train/test split."""

    set_seed(seed)
    adj, features, labels = _load_graph_file(resolve_dataset_path(dataset, file_path=file_path))

    all_idx = np.arange(labels.shape[0], dtype=int)
    normal_idx = all_idx[labels == 0].copy()
    anomaly_idx = all_idx[labels == 1].copy()
    rng = np.random.default_rng(seed)
    rng.shuffle(normal_idx)
    rng.shuffle(anomaly_idx)

    n_train = min(len(normal_idx), int(len(normal_idx) * train_rate))
    if train_rate > 0 and n_train < 1 and len(normal_idx) > 0:
        n_train = 1
    idx_train = normal_idx[:n_train].copy()
    idx_test = np.concatenate([normal_idx[n_train:], anomaly_idx]).astype(int)
    rng.shuffle(idx_train)
    rng.shuffle(idx_test)
    normal_label_idx = idx_train[: int(len(idx_train) * 0.5)].copy()
    rng.shuffle(normal_label_idx)

    if idx_train.size == 0:
        raise ValueError(f"{dataset}: strict split produced no training nodes")
    if np.any(labels[idx_train] != 0):
        raise ValueError(f"{dataset}: strict protocol violation; train split contains anomaly labels")

    adj = sp.csr_matrix(adj).astype(np.float32).tolil()
    adj.setdiag(0)
    adj = adj.tocsr()
    adj.eliminate_zeros()
    features = sp.csr_matrix(features).astype(np.float32)

    return StrictData(
        dataset=dataset,
        seed=seed,
        adj=adj,
        features=features,
        labels=labels,
        idx_train=idx_train,
        idx_test=idx_test,
        normal_label_idx=normal_label_idx,
    )


def _row_normalized_adj(adj: sp.csr_matrix) -> sp.csr_matrix:
    degree = np.asarray(adj.sum(axis=1)).reshape(-1).astype(np.float64)
    inv_degree = np.zeros_like(degree)
    mask = degree > 0
    inv_degree[mask] = 1.0 / degree[mask]
    return sp.diags(inv_degree).dot(adj).tocsr()


def _sparse_row_l2(x: sp.spmatrix) -> np.ndarray:
    return np.sqrt(np.asarray(x.multiply(x).sum(axis=1)).reshape(-1))


def build_na_features(data: StrictData, include_two_hop: bool = True) -> sp.csr_matrix:
    """Build neighborhood-aggregation features without using labels.

    Features are [X, A_row X, A_row^2 X, graph_stats]. This follows the strong
    classical baseline pattern from recent GAD benchmarking while preserving
    the strict label-visibility protocol.
    """

    x = normalize(data.features, norm="l2", axis=1, copy=True)
    row_adj = _row_normalized_adj(data.adj)
    neigh1 = row_adj.dot(x).tocsr()
    blocks = [x.tocsr(), neigh1]
    if include_two_hop:
        blocks.append(row_adj.dot(neigh1).tocsr())

    degree = np.asarray(data.adj.sum(axis=1)).reshape(-1).astype(np.float64)
    log_degree = np.log1p(degree)
    train_log_degree = log_degree[data.idx_train]
    scale = np.percentile(train_log_degree, 75) - np.percentile(train_log_degree, 25)
    if not np.isfinite(scale) or scale < 1e-12:
        scale = np.std(train_log_degree) + 1e-12
    degree_z = (log_degree - np.median(train_log_degree)) / scale

    residual = x - neigh1
    residual_l2 = _sparse_row_l2(residual)
    feature_l2 = _sparse_row_l2(x)
    neigh_l2 = _sparse_row_l2(neigh1)
    graph_stats = np.column_stack(
        [
            log_degree,
            degree_z,
            np.abs(degree_z),
            residual_l2,
            feature_l2,
            neigh_l2,
        ]
    ).astype(np.float32)
    blocks.append(sp.csr_matrix(graph_stats))

    return sp.hstack(blocks, format="csr").astype(np.float32)


def precision_at_k(scores: np.ndarray, labels: np.ndarray, k: int | None = None) -> float:
    scores = np.asarray(scores).reshape(-1)
    labels = np.asarray(labels).reshape(-1).astype(int)
    finite = np.isfinite(scores)
    scores = scores[finite]
    labels = labels[finite]
    if k is None:
        k = int(np.sum(labels == 1))
    k = min(k, len(scores))
    if k <= 0:
        return 0.0
    top_k = np.argsort(scores)[::-1][:k]
    return float(np.mean(labels[top_k] == 1))


def evaluate_scores(scores: np.ndarray, labels: np.ndarray, idx_test: Iterable[int]) -> dict[str, float]:
    idx_test = np.asarray(list(idx_test), dtype=int)
    y_true = np.asarray(labels).reshape(-1).astype(int)[idx_test]
    y_score = np.asarray(scores).reshape(-1)[idx_test]
    finite = np.isfinite(y_score)
    y_true = y_true[finite]
    y_score = y_score[finite]
    if len(np.unique(y_true)) < 2:
        return {"auc": 0.0, "ap": 0.0, "precision_at_k": 0.0}
    return {
        "auc": float(roc_auc_score(y_true, y_score)),
        "ap": float(average_precision_score(y_true, y_score)),
        "precision_at_k": precision_at_k(y_score, y_true),
    }

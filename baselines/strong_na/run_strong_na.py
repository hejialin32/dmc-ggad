"""Run strong classical NA baselines under the strict normal-only protocol."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np
from scipy import sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import warnings

from strict_protocol import (
    LABEL_VISIBILITY,
    STRICT_PROTOCOL_NAME,
    build_na_features,
    evaluate_scores,
    load_strict_data,
)


DEFAULT_DATASETS = [
    "facebook",
    "cora",
    "photo",
    "acm",
    "citeseer",
    "dblp",
    "flickr",
    "tolokers",
    "weibo",
    "pubmed",
]
DEFAULT_SEEDS = [0, 1, 3, 5, 42, 123, 456, 789, 3407, 2026]
DEFAULT_METHODS = ["xgboost_na", "rf_na", "extratrees_na", "mlp_na"]
FIELDNAMES = [
    "dataset",
    "method",
    "seed",
    "status",
    "auc",
    "ap",
    "precision_at_k",
    "seconds",
    "protocol",
    "label_visibility",
    "feature_builder",
    "feature_fit_scope",
    "train_rate",
    "n_train_normal",
    "n_pseudo_anomaly",
    "n_test",
    "n_test_anomaly",
    "svd_dim",
    "pseudo_ratio",
    "pseudo_noise",
    "n_estimators",
    "max_iter",
    "error",
]


def parse_csv_list(text: str | None, default: list[str]) -> list[str]:
    if not text:
        return list(default)
    return [item.strip() for item in text.split(",") if item.strip()]


def parse_seed_list(text: str | None) -> list[int]:
    if not text:
        return list(DEFAULT_SEEDS)
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def reduce_features(x: sp.spmatrix, svd_dim: int, seed: int) -> np.ndarray:
    """Fit an unsupervised SVD on visible node attributes/structure features."""

    max_components = min(x.shape[0] - 1, x.shape[1] - 1)
    if svd_dim > 0 and max_components > 1 and x.shape[1] > svd_dim:
        n_components = min(svd_dim, max_components)
        svd = TruncatedSVD(n_components=n_components, random_state=seed)
        return svd.fit_transform(x).astype(np.float32)
    if sp.issparse(x):
        return x.toarray().astype(np.float32)
    return np.asarray(x, dtype=np.float32)


def make_pseudo_anomalies(x_train: np.ndarray, seed: int, ratio: float, noise: float) -> np.ndarray:
    """Generate synthetic negatives from normal training nodes only."""

    rng = np.random.default_rng(seed)
    n_train, n_features = x_train.shape
    n_pseudo = max(1, int(round(n_train * ratio)))
    base = x_train[rng.integers(0, n_train, size=n_pseudo)]
    peer = x_train[rng.integers(0, n_train, size=n_pseudo)]
    perm = rng.permutation(n_features)
    pseudo = 0.7 * base[:, perm] + 0.3 * peer
    if noise > 0:
        scale = np.std(x_train, axis=0, keepdims=True)
        scale = np.where(np.isfinite(scale), scale, 0.0)
        pseudo = pseudo + rng.normal(0.0, noise, size=pseudo.shape).astype(np.float32) * (scale + 1e-6)
    return pseudo.astype(np.float32)


def build_estimator(method: str, seed: int, n_estimators: int, max_iter: int):
    method = method.lower()
    if method == "rf_na":
        return RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=None,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        )
    if method == "extratrees_na":
        return ExtraTreesClassifier(
            n_estimators=n_estimators,
            max_depth=None,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
    if method == "mlp_na":
        return make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(128,),
                activation="relu",
                alpha=1e-4,
                batch_size=256,
                learning_rate_init=1e-3,
                max_iter=max_iter,
                early_stopping=False,
                random_state=seed,
            ),
        )
    if method == "xgboost_na":
        try:
            from xgboost import XGBClassifier
        except Exception as exc:  # pragma: no cover - depends on environment
            raise RuntimeError("xgboost is not installed") from exc
        return XGBClassifier(
            n_estimators=n_estimators,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=seed,
            n_jobs=-1,
        )
    raise ValueError(f"Unknown method: {method}")


def anomaly_scores(estimator, x: np.ndarray) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(x)
        if isinstance(proba, list):
            proba = proba[0]
        return np.asarray(proba)[:, 1]
    if hasattr(estimator, "decision_function"):
        return np.asarray(estimator.decision_function(x)).reshape(-1)
    return np.asarray(estimator.predict(x)).reshape(-1)


def completed_keys(path: Path) -> set[tuple[str, str, int]]:
    if not path.exists():
        return set()
    keys: set[tuple[str, str, int]] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") == "done":
                keys.add((row["dataset"], row["method"], int(row["seed"])))
    return keys


def append_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow({name: row.get(name, "") for name in FIELDNAMES})
        handle.flush()
        os.fsync(handle.fileno())


def run_case(args, dataset: str, method: str, seed: int) -> dict:
    started = time.time()
    base = {
        "dataset": dataset,
        "method": method,
        "seed": seed,
        "protocol": STRICT_PROTOCOL_NAME,
        "label_visibility": LABEL_VISIBILITY,
        "feature_builder": "raw+1hop_mean+2hop_mean+graph_stats",
        "feature_fit_scope": "all_visible_graph_attributes_no_labels",
        "train_rate": args.train_rate,
        "svd_dim": args.svd_dim,
        "pseudo_ratio": args.pseudo_ratio,
        "pseudo_noise": args.pseudo_noise,
        "n_estimators": args.n_estimators,
        "max_iter": args.max_iter,
    }
    try:
        data = load_strict_data(dataset, seed=seed, train_rate=args.train_rate, file_path=args.file_path)
        x_sparse = build_na_features(data, include_two_hop=not args.no_two_hop)
        x_all = reduce_features(x_sparse, svd_dim=args.svd_dim, seed=seed)
        x_normal = x_all[data.idx_train]
        x_pseudo = make_pseudo_anomalies(
            x_normal,
            seed=seed + 1009,
            ratio=args.pseudo_ratio,
            noise=args.pseudo_noise,
        )
        x_train = np.vstack([x_normal, x_pseudo]).astype(np.float32)
        y_train = np.concatenate(
            [np.zeros(x_normal.shape[0], dtype=int), np.ones(x_pseudo.shape[0], dtype=int)]
        )

        estimator = build_estimator(method, seed=seed, n_estimators=args.n_estimators, max_iter=args.max_iter)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            estimator.fit(x_train, y_train)
        scores = anomaly_scores(estimator, x_all)
        metrics = evaluate_scores(scores, data.labels, data.idx_test)
        elapsed = time.time() - started
        return {
            **base,
            "status": "done",
            "auc": f"{metrics['auc']:.8f}",
            "ap": f"{metrics['ap']:.8f}",
            "precision_at_k": f"{metrics['precision_at_k']:.8f}",
            "seconds": f"{elapsed:.2f}",
            "n_train_normal": int(len(data.idx_train)),
            "n_pseudo_anomaly": int(len(x_pseudo)),
            "n_test": int(len(data.idx_test)),
            "n_test_anomaly": int(np.sum(data.labels[data.idx_test] == 1)),
            "error": "",
        }
    except Exception as exc:
        elapsed = time.time() - started
        error = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        if args.debug:
            error = traceback.format_exc()
        return {
            **base,
            "status": "failed",
            "auc": "",
            "ap": "",
            "precision_at_k": "",
            "seconds": f"{elapsed:.2f}",
            "n_train_normal": "",
            "n_pseudo_anomaly": "",
            "n_test": "",
            "n_test_anomaly": "",
            "error": error,
        }


def write_manifest(args, output_csv: Path) -> None:
    manifest = {
        "protocol": STRICT_PROTOCOL_NAME,
        "label_visibility": LABEL_VISIBILITY,
        "datasets": parse_csv_list(args.datasets, DEFAULT_DATASETS),
        "methods": parse_csv_list(args.methods, DEFAULT_METHODS),
        "seeds": parse_seed_list(args.seeds),
        "output_csv": str(output_csv),
        "notes": [
            "No validation/test labels are used for fitting SVD, pseudo-negative generation, training, or model selection.",
            "Test labels are used only once for AUROC/AP/Precision@K reporting.",
        ],
    }
    output_csv.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", type=str, default=None, help="Comma-separated datasets.")
    parser.add_argument("--methods", type=str, default=None, help="Comma-separated methods.")
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated seeds.")
    parser.add_argument("--file_path", type=str, default=None, help="Optional dataset file or directory.")
    parser.add_argument("--output_csv", type=str, default="baselines/strong_na/strong_na_results.csv")
    parser.add_argument("--train_rate", type=float, default=0.25)
    parser.add_argument("--svd_dim", type=int, default=256)
    parser.add_argument("--pseudo_ratio", type=float, default=1.0)
    parser.add_argument("--pseudo_noise", type=float, default=0.05)
    parser.add_argument("--n_estimators", type=int, default=300)
    parser.add_argument("--max_iter", type=int, default=200)
    parser.add_argument("--no_two_hop", action="store_true")
    parser.add_argument("--skip_completed", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    datasets = parse_csv_list(args.datasets, DEFAULT_DATASETS)
    methods = parse_csv_list(args.methods, DEFAULT_METHODS)
    seeds = parse_seed_list(args.seeds)
    output_csv = Path(args.output_csv)
    done = completed_keys(output_csv) if args.skip_completed else set()
    write_manifest(args, output_csv)

    for dataset in datasets:
        for seed in seeds:
            for method in methods:
                key = (dataset, method, seed)
                if key in done:
                    print(f"[skip] {dataset} {method} seed={seed}")
                    continue
                print(f"[run] dataset={dataset} method={method} seed={seed}", flush=True)
                row = run_case(args, dataset=dataset, method=method, seed=seed)
                append_row(output_csv, row)
                print(
                    f"[{row['status']}] dataset={dataset} method={method} seed={seed} "
                    f"auc={row.get('auc', '')} ap={row.get('ap', '')} error={row.get('error', '')}",
                    flush=True,
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

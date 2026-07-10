import argparse
import json
import os
import random
import sys
import time
import types
from pathlib import Path

import dgl
import networkx.algorithms.community as nx_comm
import numpy as np
import pandas as pd
import torch
from dgl.data.utils import load_graphs
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

# Flex-GAD imports umap for an optional visualizer, but the training path here
# does not use it. Keep the official model importable when umap-learn is absent.
sys.modules.setdefault("umap", types.SimpleNamespace())

from auto_encoder import GNNStructEncoder


DATASETS = [
    "reddit",
    "weibo",
    "amazon",
    "yelp",
    "tfinance",
    "elliptic",
    "tolokers",
    "questions",
]
SEEDS = list(range(3407, 10000, 10))


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def parse_datasets(value):
    if value == "all":
        return DATASETS
    out = []
    for item in value.split(","):
        item = item.strip()
        if item.isdigit():
            out.append(DATASETS[int(item)])
        elif "-" in item and all(part.isdigit() for part in item.split("-", 1)):
            start, end = item.split("-", 1)
            out.extend(DATASETS[int(start): int(end) + 1])
        elif item:
            out.append(item)
    return out


def normalize_features(x):
    x = x.float()
    xmin = x.min()
    xmax = x.max()
    denom = xmax - xmin
    if float(denom.abs().item()) < 1e-12:
        return x * 0.0
    return (x - xmin) / denom


def dgl_to_pyg(g):
    src, dst = g.edges()
    edge_index = torch.stack([src.long(), dst.long()], dim=0)
    x = normalize_features(g.ndata["feature"])
    y = g.ndata["label"].long()
    return Data(x=x, edge_index=edge_index, num_nodes=g.num_nodes(), y=y)


def build_sparse_norm_adj(edge_index, num_nodes, device):
    loops = torch.arange(num_nodes, dtype=torch.long)
    loops = torch.stack([loops, loops], dim=0)
    ei = torch.cat([edge_index.cpu(), loops], dim=1)
    values = torch.ones(ei.shape[1], dtype=torch.float32)
    deg = torch.zeros(num_nodes, dtype=torch.float32)
    deg.scatter_add_(0, ei[0], values)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0
    norm_values = deg_inv_sqrt[ei[0]] * values * deg_inv_sqrt[ei[1]]
    return torch.sparse_coo_tensor(ei, norm_values, (num_nodes, num_nodes)).coalesce().to(device)


def build_neighbor_info(edge_index, num_nodes, device):
    neighbor_dict = {i: [] for i in range(num_nodes)}
    src = edge_index[0].cpu().tolist()
    dst = edge_index[1].cpu().tolist()
    for s, d in zip(src, dst):
        neighbor_dict[s].append(d)
    neighbor_num_list = torch.tensor(
        [len(neighbor_dict[i]) for i in range(num_nodes)], dtype=torch.long
    ).to(device)
    return neighbor_dict, neighbor_num_list


def build_community_features(data, seed):
    # Flex-GAD uses Louvain community means as a second feature view.
    # Keep the same idea, but isolate it here so failures are visible.
    graph = to_networkx(data, to_undirected=True)
    communities = nx_comm.louvain_communities(graph, seed=seed)
    x_new = torch.zeros_like(data.x)
    for community in communities:
        nodes = torch.tensor(list(community), dtype=torch.long)
        mean = data.x[nodes].mean(dim=0)
        norm = mean.norm(p=2)
        if float(norm.item()) > 1e-12:
            mean = mean / norm
        x_new[nodes] = mean
    return x_new


def precision_recall_at_k(labels, scores):
    k = int(labels.sum())
    if k <= 0:
        return float("nan"), float("nan")
    k = min(k, labels.shape[0])
    top = np.argsort(-scores)[:k]
    hits = labels[top].sum()
    return float(hits / k), float(hits / max(labels.sum(), 1))


def evaluate(labels, scores, mask):
    labels = labels.detach().cpu().numpy().astype(int)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    mask = mask.detach().cpu().numpy().astype(bool)
    y = labels[mask]
    s = scores[mask]
    if len(np.unique(y)) < 2:
        return {
            "AUROC": float("nan"),
            "AUPRC": float("nan"),
            "PrecisionK": float("nan"),
            "RecallK": float("nan"),
        }
    pk, rk = precision_recall_at_k(y, s)
    return {
        "AUROC": float(roc_auc_score(y, s)),
        "AUPRC": float(average_precision_score(y, s)),
        "PrecisionK": pk,
        "RecallK": rk,
    }


def train_one(data, test_mask, args, device):
    edge_index = data.edge_index
    num_nodes = data.num_nodes

    x_new = build_community_features(data, args.seed)
    loop = torch.arange(num_nodes, dtype=torch.long)
    loop = torch.stack([loop, loop], dim=0)
    train_edge_index = torch.cat([edge_index, loop], dim=1).to(device)

    x = data.x.to(device)
    x_new = x_new.to(device)
    norm_adj = build_sparse_norm_adj(edge_index, num_nodes, device)
    neighbor_dict, neighbor_num_list = build_neighbor_info(train_edge_index, num_nodes, device)

    model = GNNStructEncoder(
        x.shape[1],
        args.dimension,
        args.dimension,
        2,
        args.sample_size,
        device=device,
        neighbor_num_list=neighbor_num_list,
        GNN_name=args.encoder,
        lambda_loss1=args.lambda_loss1,
        lambda_loss2=args.lambda_loss2,
    ).to(device)
    degree_params = list(map(id, model.degree_decoder.parameters()))
    base_params = filter(lambda p: id(p) not in degree_params, model.parameters())
    optimizer = torch.optim.Adam(
        [
            {"params": base_params},
            {"params": model.degree_decoder.parameters(), "lr": 1e-2},
        ],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    final_scores = None
    best_test_auc = -1.0
    best_test_scores = None
    labels = data.y.cpu()

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        (
            loss,
            loss_per_node,
            h_loss_per_node,
            feature_loss_per_node,
            _h_loss,
            _feature_loss,
            attn_importance,
        ) = model(train_edge_index, x, neighbor_num_list, neighbor_dict, norm_adj, norm_adj, x_new, device)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        with torch.no_grad():
            h_loss_per_node = h_loss_per_node.detach().reshape(-1).cpu()
            feature_loss_per_node = feature_loss_per_node.detach().reshape(-1).cpu()
            h_denom = max(float((h_loss_per_node.max() - h_loss_per_node.min()).item()), 1e-12)
            f_denom = max(float((feature_loss_per_node.max() - feature_loss_per_node.min()).item()), 1e-12)
            h_norm = (h_loss_per_node - h_loss_per_node.min()) / h_denom
            f_norm = (feature_loss_per_node - feature_loss_per_node.min()) / f_denom
            h_weight, f_weight = [float(v) for v in attn_importance.detach().cpu().reshape(-1)[:2]]
            final_scores = (h_weight * h_norm + f_weight * f_norm).numpy()
            metrics = evaluate(labels, final_scores, test_mask)
            if metrics["AUROC"] == metrics["AUROC"] and metrics["AUROC"] > best_test_auc:
                best_test_auc = metrics["AUROC"]
                best_test_scores = final_scores.copy()

    strict_metrics = evaluate(labels, final_scores, test_mask)
    best_metrics = evaluate(labels, best_test_scores, test_mask) if best_test_scores is not None else strict_metrics
    return strict_metrics, best_metrics


def append_status(path, row):
    payload = dict(row)
    payload["error"] = json.dumps(str(row.get("error", "")), ensure_ascii=False)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(
            "{time}\t{method}\t{dataset}\t{trial}\t{status}\t{seconds:.2f}\t"
            "{AUROC}\t{AUPRC}\t{PrecisionK}\t{RecallK}\t{error}\n".format(**payload)
        )
        fh.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="reddit,tolokers,weibo,questions")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--lambda_loss1", type=float, default=0.9)
    parser.add_argument("--lambda_loss2", type=float, default=0.25)
    parser.add_argument("--sample_size", type=int, default=10)
    parser.add_argument("--dimension", type=int, default=128)
    parser.add_argument("--encoder", default="SeriesEncoder")
    parser.add_argument("--weight_decay", type=float, default=3e-4)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-root", default="/root/GADBench/datasets")
    parser.add_argument("--out", default="/root/Flex-GAD/results/flexgad_gadbench_v1.csv")
    parser.add_argument("--status", default="/root/Flex-GAD/results/flexgad_gadbench_v1_status.tsv")
    parser.add_argument("--max-nodes", type=int, default=60000)
    parser.add_argument("--max-edges", type=int, default=1000000)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    out_path = Path(args.out)
    status_path = Path(args.status)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not status_path.exists():
        status_path.write_text(
            "time\tmethod\tdataset\ttrial\tstatus\tseconds\tAUROC\tAUPRC\tPrecisionK\tRecallK\terror\n",
            encoding="utf-8",
        )

    rows = []
    if out_path.exists():
        rows = pd.read_csv(out_path).to_dict("records")
    done = {
        (str(r["dataset"]), int(r["trial"]))
        for r in rows
        if str(r.get("status")) == "done"
    }

    datasets = parse_datasets(args.datasets)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    for dataset in datasets:
        graph = load_graphs(str(Path(args.data_root) / dataset))[0][0]
        if graph.num_nodes() > args.max_nodes or graph.num_edges() > args.max_edges:
            for trial in range(args.trials):
                key = (dataset, trial)
                if key in done:
                    continue
                row = {
                    "method": "Flex-GAD",
                    "dataset": dataset,
                    "trial": trial,
                    "seed": SEEDS[trial],
                    "status": "skipped",
                    "seconds": 0.0,
                    "AUROC": np.nan,
                    "AUPRC": np.nan,
                    "PrecisionK": np.nan,
                    "RecallK": np.nan,
                    "best_AUROC_reference": np.nan,
                    "best_AUPRC_reference": np.nan,
                    "error": f"graph too large for Flex-GAD queue: nodes={graph.num_nodes()} edges={graph.num_edges()}",
                }
                rows.append(row)
                pd.DataFrame(rows).to_csv(out_path, index=False)
                append_status(status_path, row | {"time": time.strftime("%Y-%m-%d %H:%M:%S")})
            continue

        data = dgl_to_pyg(graph)
        for trial in range(args.trials):
            key = (dataset, trial)
            if key in done:
                continue
            start = time.time()
            stamp = time.strftime("%Y-%m-%d %H:%M:%S")
            try:
                seed = SEEDS[trial]
                args.seed = seed
                set_seed(seed)
                test_mask = graph.ndata["test_masks"][:, trial].bool()
                strict, best_ref = train_one(data, test_mask, args, device)
                seconds = time.time() - start
                row = {
                    "method": "Flex-GAD",
                    "dataset": dataset,
                    "trial": trial,
                    "seed": seed,
                    "status": "done",
                    "seconds": seconds,
                    "AUROC": strict["AUROC"],
                    "AUPRC": strict["AUPRC"],
                    "PrecisionK": strict["PrecisionK"],
                    "RecallK": strict["RecallK"],
                    "best_AUROC_reference": best_ref["AUROC"],
                    "best_AUPRC_reference": best_ref["AUPRC"],
                    "error": "",
                }
            except Exception as exc:
                seconds = time.time() - start
                row = {
                    "method": "Flex-GAD",
                    "dataset": dataset,
                    "trial": trial,
                    "seed": SEEDS[trial],
                    "status": "failed",
                    "seconds": seconds,
                    "AUROC": np.nan,
                    "AUPRC": np.nan,
                    "PrecisionK": np.nan,
                    "RecallK": np.nan,
                    "best_AUROC_reference": np.nan,
                    "best_AUPRC_reference": np.nan,
                    "error": repr(exc)[:1000],
                }
            rows.append(row)
            pd.DataFrame(rows).to_csv(out_path, index=False)
            append_status(status_path, row | {"time": stamp})


if __name__ == "__main__":
    main()

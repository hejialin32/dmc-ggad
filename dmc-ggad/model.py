import random

import torch
import torch.nn as nn
import torch.nn.functional as F


class DensityAwareStructuralGenerator(nn.Module):
    def __init__(self, hidden_dim, global_sample_rate=0.1, hop=2):
        super().__init__()
        self.hop = hop
        self.global_sample_rate = global_sample_rate
        self.struct_att = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self._adj_list = None
        self._neighbor_cache = {}
        self._density_cache = {}
        self._total_edges = None

    def _build_adj_list(self, adj):
        if self._adj_list is not None:
            return

        num_nodes = adj.shape[0]
        self._adj_list = [[] for _ in range(num_nodes)]
        if adj.is_sparse:
            rows, cols = adj.coalesce().indices()
        else:
            rows, cols = torch.nonzero(adj, as_tuple=True)

        rows = rows.detach().cpu().tolist()
        cols = cols.detach().cpu().tolist()
        for row, col in zip(rows, cols):
            self._adj_list[row].append(col)
        self._total_edges = len(rows)

    def get_multi_hop_neighbors(self, node_idx, adj):
        cache_key = (int(node_idx), self.hop)
        if cache_key in self._neighbor_cache:
            return self._neighbor_cache[cache_key]

        self._build_adj_list(adj)
        neighbors = set()
        frontier = {int(node_idx)}
        for _ in range(self.hop):
            next_frontier = set()
            for node in frontier:
                for neighbor in self._adj_list[node]:
                    if neighbor != node_idx and neighbor not in neighbors:
                        neighbors.add(neighbor)
                        next_frontier.add(neighbor)
            frontier = next_frontier
            if not frontier:
                break

        result = list(neighbors)
        self._neighbor_cache[cache_key] = result
        return result

    def calculate_neighbor_density(self, node_idx, adj):
        node_idx = int(node_idx)
        if node_idx in self._density_cache:
            return self._density_cache[node_idx]

        self._build_adj_list(adj)
        average_degree = self._total_edges / max(1, len(self._adj_list))
        density = len(self._adj_list[node_idx]) / (average_degree + 1e-6)
        self._density_cache[node_idx] = density
        return density

    def forward(self, all_emb, target_idx, adj, sample_k=10):
        device = all_emb.device
        num_nodes = all_emb.shape[0]
        global_sample_size = max(10, int(num_nodes * self.global_sample_rate))
        generated = []

        for idx in target_idx:
            idx = int(idx)
            if self.calculate_neighbor_density(idx, adj) > 1.0:
                candidates = self.get_multi_hop_neighbors(idx, adj)
                global_sample = torch.randperm(num_nodes)[:global_sample_size].tolist()
                candidates = list(set(candidates + global_sample))
                if len(candidates) > sample_k * 3:
                    candidates = random.sample(candidates, sample_k * 3)
                if not candidates:
                    candidates = torch.randint(0, num_nodes, (sample_k,), device=device)
                else:
                    candidates = torch.tensor(candidates, device=device)

                curr_emb = all_emb[idx].unsqueeze(0).repeat(len(candidates), 1)
                cand_emb = all_emb[candidates]
                att_input = torch.cat([curr_emb, cand_emb, torch.randn_like(curr_emb)], dim=1)
                weights = self.struct_att(att_input)
                new_emb = torch.sum(weights * cand_emb, dim=0, keepdim=True)
            else:
                neighbors = self.get_multi_hop_neighbors(idx, adj)
                random_nodes = torch.randint(0, num_nodes, (sample_k,), device=device).tolist()
                candidates = list(set(neighbors + random_nodes)) if neighbors else random_nodes
                if len(candidates) > sample_k * 3:
                    candidates = random.sample(candidates, sample_k * 3)
                candidates = torch.tensor(candidates, device=device)
                weights = F.softmax(torch.rand(len(candidates), 1, device=device), dim=0)
                new_emb = torch.sum(weights * all_emb[candidates], dim=0, keepdim=True)

            generated.append(new_emb)

        return torch.cat(generated, dim=0)


class GCN(nn.Module):
    def __init__(self, in_dim, out_dim, activation='prelu', bias=True):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.act = nn.PReLU() if activation == 'prelu' else activation
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))
        else:
            self.register_parameter('bias', None)
        nn.init.xavier_uniform_(self.fc.weight.data)

    def forward(self, x, adj, sparse=True):
        x = self.fc(x)
        if sparse:
            out = torch.spmm(adj, x.squeeze(0)).unsqueeze(0)
        else:
            out = torch.bmm(adj, x)
        if self.bias is not None:
            out = out + self.bias
        return self.act(out)


class Model(nn.Module):
    def __init__(
        self,
        n_in,
        n_h,
        activation='prelu',
        global_sample_rate=0.1,
        hop=2,
        num_layers=3,
    ):
        super().__init__()
        self.gcn_layers = nn.ModuleList([GCN(n_in, n_h, activation)])
        self.gcn_layers.extend(GCN(n_h, n_h, activation) for _ in range(num_layers - 1))
        self.generator = DensityAwareStructuralGenerator(
            n_h,
            global_sample_rate=global_sample_rate,
            hop=hop,
        )
        self.fc1 = nn.Linear(n_h, n_h // 2, bias=False)
        self.fc2 = nn.Linear(n_h // 2, n_h // 4, bias=False)
        self.fc3 = nn.Linear(n_h // 4, 1, bias=False)
        self.act = nn.ReLU()

    def score(self, emb):
        x = self.act(self.fc1(emb))
        x = self.act(self.fc2(x))
        return self.fc3(x)

    def forward(self, features, adj, abnormal_idx, normal_idx, train_flag, args, sparse=True):
        emb = features
        for gcn in self.gcn_layers:
            emb = gcn(emb, adj, sparse)

        emb_con = None
        emb_abnormal = emb[:, abnormal_idx, :]
        noise = torch.randn_like(emb_abnormal) * args.var + args.mean
        emb_abnormal = emb_abnormal + noise

        if train_flag:
            emb_con = self.generator(emb[0], abnormal_idx, adj)
            emb_combine = torch.cat((emb[:, normal_idx, :], emb_con.unsqueeze(0)), dim=1)
            logits = self.score(emb_combine)
            emb[:, abnormal_idx, :] = emb_con
        else:
            emb_combine = None
            logits = self.score(emb)

        return emb, emb_combine, logits, emb_con, emb_abnormal

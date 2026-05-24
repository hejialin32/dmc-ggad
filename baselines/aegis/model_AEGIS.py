import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.nn import MLP

import numpy as np

def neighList_to_edgeList(adj):
    edge_list = []
    for i in range(adj.shape[0]):
        for j in torch.argwhere(adj[i, :] >0):
            edge_list.append([int(i), int(j)])
    return edge_list

def neighList_to_edgeList_train(adj, idx_train):
    edge_list = []
    for i in idx_train:
        for j in torch.argwhere(adj[i, :] >0):
            edge_list.append([int(i), int(j)])
    return edge_list

class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, act, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU() if act == 'prelu' else act

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq, adj, sparse=False):
        seq_fts = self.fc(seq)
        if sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
        else:
            out = torch.mm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias

        return self.act(out)


class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq):
        return torch.mean(seq, 1)


class MaxReadout(nn.Module):
    def __init__(self):
        super(MaxReadout, self).__init__()

    def forward(self, seq):
        return torch.max(seq, 1).values


class MinReadout(nn.Module):
    def __init__(self):
        super(MinReadout, self).__init__()

    def forward(self, seq):
        return torch.min(seq, 1).values


class WSReadout(nn.Module):
    def __init__(self):
        super(WSReadout, self).__init__()

    def forward(self, seq, query):
        query = query.permute(0, 2, 1)
        sim = torch.matmul(seq, query)
        sim = F.softmax(sim, dim=1)
        sim = sim.repeat(1, 1, 64)
        out = torch.mul(seq, sim)
        out = torch.sum(out, 1)
        return out


class Discriminator(nn.Module):
    def __init__(self, n_h, negsamp_round):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

        self.negsamp_round = negsamp_round

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl):
        scs = []
        # positive
        scs.append(self.f_k(h_pl, c))

        # negative
        c_mi = c
        for _ in range(self.negsamp_round):
            c_mi = torch.cat((c_mi[-2:-1, :], c_mi[:-1, :]), 0)
            scs.append(self.f_k(h_pl, c_mi))

        logits = torch.cat(tuple(scs))

        return logits


class Model(nn.Module):
    def __init__(self, n_in, n_h, activation, negsamp_round, readout):
        super(Model, self).__init__()
        self.noise_dim = 16
        self.hid_dim = 64
        self.read_mode = readout

        self.act = nn.ReLU()
        if readout == 'max':
            self.read = MaxReadout()
        elif readout == 'min':
            self.read = MinReadout()
        elif readout == 'avg':
            self.read = AvgReadout()
        elif readout == 'weighted_sum':
            self.read = WSReadout()

        self.disc = Discriminator(n_h, negsamp_round)

        noise_dim = 16
        hid_dim = 64
        num_layers = 4
        dropout = 0.
        in_dim = n_in
        generator_layers = math.floor(num_layers / 2)
        encoder_layers = math.ceil(num_layers / 2)
        act = torch.nn.functional.relu
        self.gcn_enc1 = GCN(n_in, n_h, activation)
        self.gcn_enc2 = GCN(n_h, n_h, activation)
        self.gcn_dec1 = GCN(n_h, n_h, activation)
        self.gcn_dec2 = GCN(n_h, n_in, activation)
        self.generator = MLP(in_channels=noise_dim,
                             hidden_channels=hid_dim,
                             out_channels=in_dim,
                             num_layers=generator_layers,
                             dropout=dropout,
                             act=act)

        self.discriminator = MLP(in_channels=in_dim,
                                 hidden_channels=hid_dim,
                                 out_channels=hid_dim,
                                 num_layers=encoder_layers,
                                 dropout=dropout,
                                 act=act
                                 )
        self.discriminator2 = nn.Sequential(
            nn.Linear(n_h, hid_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim, 1)
        )



    def model_enc(self, x, adj,  noise, idx_train, sparse=False):
        """
        Forward computation.

        Parameters
        ----------
        x : torch.Tensor
            Input attribute embeddings.
        noise : torch.Tensor
            Input noise.

        Returns
        -------
        x_ : torch.Tensor
            Reconstructed node features.
        a : torch.Tensor
            Reconstructed adjacency matrix from real samples.
        a_ : torch.Tensor
            Reconstructed adjacency matrix from fake samples.
        """
        x_gen = self.generator(noise.to(x.device))
        # x_gen = self.generator(noise)
        z_gen = self.gcn_enc1(x_gen, adj, sparse)
        z_gen = self.gcn_enc2(z_gen, adj, sparse)

        z = self.gcn_enc1(x, adj, sparse)
        z = self.gcn_enc2(z, adj, sparse)

        z_gen_dec = self.gcn_dec1(z, adj, sparse)
        z_gen_dec = self.gcn_dec2(z_gen_dec, adj, sparse)

        z_dec = self.gcn_dec1(z, adj, sparse)
        z_dec = self.gcn_dec2(z_dec, adj, sparse)

        emb_all = torch.cat([z, z_gen], 0)

        # 根据张量维度决定处理方式
        if emb_all.dim() == 3:
            # 3D张量: [batch, nodes, features] - 用于elliptic等数据集
            label = torch.cat([torch.zeros(z.shape[1]), torch.ones(z_gen.shape[1])])
            emb_all_2d = emb_all.reshape(-1, emb_all.shape[-1])

            logits = torch.nan_to_num(self.discriminator2(emb_all_2d), nan=0.0, posinf=20.0, neginf=-20.0)
            logits_gen = torch.nan_to_num(self.discriminator2(z_gen.reshape(-1, z_gen.shape[-1])), nan=0.0, posinf=20.0, neginf=-20.0)

            idx_train_expanded = list(idx_train) + [z.shape[1]+i for i in range(len(idx_train))]
            loss_dis = F.binary_cross_entropy_with_logits(logits[idx_train_expanded, 0], label[idx_train_expanded].to(x.device))
        else:
            # 2D张量: [nodes, features] - 用于其他数据集
            label = torch.cat([torch.zeros(len(z)), torch.ones(len(z_gen))])

            logits = torch.nan_to_num(self.discriminator2(emb_all), nan=0.0, posinf=20.0, neginf=-20.0)
            logits_gen = torch.nan_to_num(self.discriminator2(z_gen), nan=0.0, posinf=20.0, neginf=-20.0)

            idx_train_expanded = list(idx_train) + [len(z)+i for i in range(len(z))]
            loss_dis = F.binary_cross_entropy_with_logits(logits[idx_train_expanded, 0], label[idx_train_expanded].to(x.device))

        logits = torch.sigmoid(logits)
        # loss_dis = F.binary_cross_entropy(logits[idx_train, 0], label[idx_train])
        loss_g = F.binary_cross_entropy_with_logits(logits_gen[:, 0], torch.zeros_like(logits_gen[:, 0]))
        return z_dec, loss_dis, loss_g, logits, emb_all


    def forward(self, seq1, adj, idx_train, idx_test, sparse=False):
        seq1 = torch.squeeze(seq1)
        if sparse:
            if adj.is_sparse_csr:
                adj = adj.to_sparse_coo()
            if adj.dim() == 3:
                adj = adj.coalesce()
                adj = torch.sparse_coo_tensor(adj.indices()[1:], adj.values(), adj.shape[1:], device=adj.device)
            else:
                adj = adj.coalesce()
        else:
            adj = torch.squeeze(adj)
        noise = torch.randn(seq1.shape[0], self.noise_dim)
        z_dec, loss_dis, loss_g, logits, emb_all = self.model_enc(seq1, adj, noise, idx_train, sparse)

        # 优化显存: 分批计算 loss_ae (避免大图OOM)

        # 确保张量维度一致
        if seq1.dim() != z_dec.dim():
            if seq1.dim() == 2 and z_dec.dim() == 3:
                z_dec = torch.squeeze(z_dec, 0)
            elif seq1.dim() == 3 and z_dec.dim() == 2:
                z_dec = z_dec.unsqueeze(0)

        batch_size = min(1024, len(idx_train))
        total_loss = 0.0
        for i in range(0, len(idx_train), batch_size):
            batch_idx = idx_train[i:i+batch_size]
            if seq1.dim() == 3:
                # 3D张量: [batch, nodes, features] -> 使用 [:, idx, :]
                diff_attr_batch = torch.pow(seq1[:, batch_idx, :] - z_dec[:, batch_idx, :], 2)
                batch_loss = torch.mean(torch.sqrt(torch.sum(diff_attr_batch, 2) + 1e-8), 1)
            else:
                # 2D张量: [nodes, features] -> 使用 [idx, :]
                diff_attr_batch = torch.pow(seq1[batch_idx, :] - z_dec[batch_idx, :], 2)
                batch_loss = torch.mean(torch.sqrt(torch.sum(diff_attr_batch, 1) + 1e-8), 0)
            total_loss += batch_loss.sum()
            del diff_attr_batch, batch_loss
            torch.cuda.empty_cache()
        loss_ae = total_loss / len(idx_train)

        score = logits[idx_test, :]
        return loss_dis, loss_g, loss_ae,  score, emb_all

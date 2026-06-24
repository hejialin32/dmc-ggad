import torch
import torch.nn as nn
import torch.nn.functional as F
import random




class DensityAwareStructuralGenerator(nn.Module):

    def __init__(self, n_h, global_sample_rate=0.1):
        super(DensityAwareStructuralGenerator, self).__init__()
        self.n_h = n_h
        self.hop = 2
        self.global_sample_rate = global_sample_rate
        self.density_mode = 'auto'
        self.disable_global_sampling = False


        self.struct_att = nn.Sequential(
            nn.Linear(n_h * 3, n_h),
            nn.ReLU(),
            nn.Linear(n_h, 1),
            nn.Sigmoid()
        )


        self.attr_gen = nn.Sequential(
            nn.Linear(n_h * 3, n_h),
            nn.ReLU(),
            nn.Linear(n_h, n_h),
            nn.Tanh()
        )


        self._adj_list = None
        self._neighbor_cache = {}
        self._density_cache = {}
        self._total_edges = None

    def _build_adj_list(self, adj):
        """Build adj list."""
        if self._adj_list is not None:
            return

        num_nodes = adj.shape[0]
        self._adj_list = [[] for _ in range(num_nodes)]

        if isinstance(adj, torch.Tensor):
            if adj.is_sparse:
                indices = adj._indices()
                rows = indices[0].tolist()
                cols = indices[1].tolist()
            else:
                rows, cols = torch.nonzero(adj, as_tuple=True)
                rows = rows.tolist()
                cols = cols.tolist()
        else:

            if hasattr(adj, 'tocoo'):
                coo = adj.tocoo()
                rows = coo.row.tolist()
                cols = coo.col.tolist()
            else:
                rows, cols = adj.nonzero()
                rows = rows.tolist()
                cols = cols.tolist()

        for r, c in zip(rows, cols):
            self._adj_list[r].append(c)

        self._total_edges = len(rows)

    def get_multi_hop_neighbors(self, node_idx, adj, hop=2):
        """Get multi hop neighbors."""
        cache_key = (node_idx, hop)
        if cache_key in self._neighbor_cache:
            return self._neighbor_cache[cache_key]

        self._build_adj_list(adj)
        neighbors = set()
        current_hop_nodes = {node_idx}

        for h in range(hop):
            next_hop_nodes = set()
            for node in current_hop_nodes:
                node_neighbors = self._adj_list[node]
                for neighbor in node_neighbors:
                    if neighbor not in neighbors and neighbor != node_idx:
                        neighbors.add(neighbor)
                        next_hop_nodes.add(neighbor)

            current_hop_nodes = next_hop_nodes
            if not current_hop_nodes:
                break

        res = list(neighbors)
        self._neighbor_cache[cache_key] = res
        return res

    def calculate_neighbor_density(self, node_idx, adj):
        """Calculate neighbor density."""
        if node_idx in self._density_cache:
            return self._density_cache[node_idx]

        self._build_adj_list(adj)
        num_neighbors = len(self._adj_list[node_idx])
        num_nodes = len(self._adj_list)

        average_degree = self._total_edges / num_nodes
        density = num_neighbors / (average_degree + 1e-6)

        self._density_cache[node_idx] = density
        return density

    def forward(self, all_emb, target_idx, adj, sample_k=10, epoch=0):
        """Forward."""
        device = all_emb.device
        generated_embs = []

        num_nodes = all_emb.shape[0]
        global_sample_size = max(10, int(num_nodes * self.global_sample_rate))

        attention_count = 0
        random_count = 0

        for idx in target_idx:

            if self.density_mode == 'auto':
                neighbor_density = self.calculate_neighbor_density(idx, adj)
                is_high_density = neighbor_density > 1.0
            elif self.density_mode in ['high', 'all_high']:
                is_high_density = True
            elif self.density_mode in ['low', 'all_low']:
                is_high_density = False
            else:
                neighbor_density = self.calculate_neighbor_density(idx, adj)
                is_high_density = neighbor_density > 1.0

            if is_high_density:
                attention_count += 1


                multi_hop_neighbors = self.get_multi_hop_neighbors(idx, adj, hop=self.hop)
                all_candidates = multi_hop_neighbors.copy()

                if not self.disable_global_sampling:
                    global_sample = torch.randperm(num_nodes)[:global_sample_size].tolist()
                    all_candidates = list(set(all_candidates + global_sample))

                if len(all_candidates) > sample_k * 3:
                    all_candidates = random.sample(all_candidates, sample_k * 3)

                if len(all_candidates) == 0:
                    candidates = torch.randint(0, num_nodes, (sample_k,), device=device)
                else:
                    candidates = torch.tensor(all_candidates, device=device)

                curr_emb = all_emb[idx].unsqueeze(0).repeat(len(candidates), 1)
                cand_embs = all_emb[candidates]
                noise = torch.randn_like(curr_emb)

                struct_input = torch.cat([curr_emb, cand_embs, noise], dim=1)
                att_weights = self.struct_att(struct_input)

                struct_context = torch.sum(att_weights * cand_embs, dim=0)



                new_emb = struct_context.unsqueeze(0)

            else:
                random_count += 1


                multi_hop_neighbors = self.get_multi_hop_neighbors(idx, adj, hop=self.hop)
                random_nodes = torch.randint(0, num_nodes, (sample_k,), device=device).tolist()

                if len(multi_hop_neighbors) > 0:
                    all_candidates = list(set(multi_hop_neighbors + random_nodes))
                else:
                    all_candidates = random_nodes

                if len(all_candidates) > sample_k * 3:
                    all_candidates = random.sample(all_candidates, sample_k * 3)

                all_candidates = torch.tensor(all_candidates, device=device)
                random_weights = torch.rand(len(all_candidates), 1, device=device)
                random_weights = F.softmax(random_weights, dim=0)

                struct_context = torch.sum(random_weights * all_emb[all_candidates], dim=0)



                new_emb = struct_context.unsqueeze(0)

            generated_embs.append(new_emb)

        if epoch == 0:
            total = attention_count + random_count
            print(f"[Generation policy] attention: {attention_count} ({attention_count/total*100:.1f}%) | "
                  f"random latent mix: {random_count} ({random_count/total*100:.1f}%)")

        return torch.cat(generated_embs, dim=0)

class GCN(nn.Module):
    """Gcn."""
    def __init__(self, in_ft, out_ft, act, bias=True):
        """Init."""
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
        """Weights init."""
        if isinstance(m, nn.Linear):

            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq, adj, sparse=True):
        """Forward."""


        seq_fts = self.fc(seq)



        if sparse:

            seq_fts = torch.squeeze(seq_fts, 0)
            adj = torch.squeeze(adj, 0)
            out = torch.spmm(adj, seq_fts)
            out = out.unsqueeze(0)
        else:
            out = torch.bmm(adj, seq_fts)



        if self.bias is not None:
            out += self.bias



        return self.act(out)


class AvgReadout(nn.Module):
    """Avgreadout."""
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq):
        """Forward."""


        return torch.mean(seq, 1)


class MaxReadout(nn.Module):
    """Maxreadout."""
    def __init__(self):
        super(MaxReadout, self).__init__()

    def forward(self, seq):
        """Forward."""
        return torch.max(seq, 1).values


class MinReadout(nn.Module):
    """Minreadout."""
    def __init__(self):
        super(MinReadout, self).__init__()

    def forward(self, seq):
        """Forward."""
        return torch.min(seq, 1).values


class WSReadout(nn.Module):
    """Wsreadout."""
    def __init__(self):
        super(WSReadout, self).__init__()

    def forward(self, seq, query):
        """Forward."""

        query = query.permute(0, 2, 1)



        sim = torch.matmul(seq, query)


        sim = F.softmax(sim, dim=1)


        sim = sim.repeat(1, 1, 64)


        out = torch.mul(seq, sim)


        out = torch.sum(out, 1)
        return out


class Discriminator(nn.Module):
    """Discriminator."""
    def __init__(self, n_h, negsamp_round):
        """Init."""
        super(Discriminator, self).__init__()




        self.f_k = nn.Bilinear(n_h, n_h, 1)


        for m in self.modules():
            self.weights_init(m)

        self.negsamp_round = negsamp_round

    def weights_init(self, m):
        """Weights init."""
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl):
        """Forward."""
        scs = []



        scs.append(self.f_k(h_pl, c))



        c_mi = c
        for _ in range(self.negsamp_round):



            c_mi = torch.cat((c_mi[-2:-1, :], c_mi[:-1, :]), 0)


            scs.append(self.f_k(h_pl, c_mi))



        logits = torch.cat(tuple(scs))
        return logits


class Model(nn.Module):
    """Model."""
    def __init__(self, n_in, n_h, activation, negsamp_round, readout, global_sample_rate=0.1, density_mode='auto', hop=2, disable_global_sampling=False, num_layers=3):
        """Init."""
        super(Model, self).__init__()
        self.read_mode = readout
        self.num_layers = num_layers



        self.gcn_layers = nn.ModuleList()

        self.gcn_layers.append(GCN(n_in, n_h, activation))

        for _ in range(num_layers - 1):
            self.gcn_layers.append(GCN(n_h, n_h, activation))




        self.fc1 = nn.Linear(n_h, int(n_h / 2), bias=False)
        self.fc2 = nn.Linear(int(n_h / 2), int(n_h / 4),
                             bias=False)
        self.fc3 = nn.Linear(int(n_h / 4), 1, bias=False)


        self.generator = DensityAwareStructuralGenerator(n_h, global_sample_rate=global_sample_rate)
        self.generator.density_mode = density_mode
        self.generator.hop = hop
        self.generator.disable_global_sampling = disable_global_sampling


        self.fc5 = nn.Linear(n_h, n_in, bias=False)

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

    def forward(self, seq1, adj, sample_abnormal_idx, normal_idx, train_flag, args, sparse=False):
        """Forward."""



        x = seq1
        for i, gcn in enumerate(self.gcn_layers):
            x = gcn(x, adj, sparse)
            if i == self.num_layers - 1:
                emb = x


        emb_con = None
        emb_combine = None
        emb_abnormal = emb[:, sample_abnormal_idx, :]





        noise = torch.randn(emb_abnormal.size()).to(emb_abnormal.device) * args.var + args.mean




        emb_abnormal = emb_abnormal + noise

        if train_flag:





            if not hasattr(self, 'current_epoch'):
                self.current_epoch = 0
            emb_con = self.generator(emb[0], sample_abnormal_idx, adj[0], epoch=self.current_epoch)


            emb_combine = torch.cat(
                (emb[:, normal_idx, :], torch.unsqueeze(emb_con, 0)), 1)






            f_1 = self.fc1(emb_combine)
            f_1 = self.act(f_1)
            f_2 = self.fc2(f_1)
            f_2 = self.act(f_2)
            f_3 = self.fc3(f_2)



            emb[:, sample_abnormal_idx, :] = emb_con

        else:


            f_1 = self.fc1(emb)
            f_1 = self.act(f_1)
            f_2 = self.fc2(f_1)
            f_2 = self.act(f_2)
            f_3 = self.fc3(f_2)


        return emb, emb_combine, f_3, emb_con, emb_abnormal






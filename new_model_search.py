from final_operation_test import *
from torch.autograd import Variable
from utils import cau_cos
from genotype import PRIMITIVES
from genotype import Genotype
from tqdm import tqdm


class MixedOp(nn.Module):  # 混合操作来求取a
    def __init__(self, nfeat, nhid, nclass, dropout, device):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        self.device = device
        for primitive in PRIMITIVES:
            op = OPS[primitive](nfeat, nhid, nclass, dropout)  # mix the operations
            self._ops.append(op)

    def forward(self, x, adj, weights):
        adj_base = adj
        _feats = []
        _adjs = []
        for op in self._ops:
            feat, adj_1 = op(x, adj)  # let the operations work
            _feats.append(feat)  # only take feature map
            _adjs.append(adj_1)
        for adj_2 in _adjs:
            adj_base = adj_base * adj_2 # make drop_edge function on every adjacent matrix

        adj_base = adj_base.to(self.device)

        return sum(w * feat for w, op, feat in
                   zip(weights, self._ops, _feats)), adj_base  # self._op can be deleted


class Block(nn.Module):
    def __init__(self, steps, nhid, nclass, dropout, nfeat, mulitplier,
                 device):
        super(Block, self).__init__()
        self._steps = steps
        self._multiplier = mulitplier
        self._ops = nn.ModuleList()
        self.device = device
        for i in range(self._steps):
            for j in range(2 + i):
                op = MixedOp(nfeat, nhid, nclass, dropout, self.device)
                self._ops.append(op)

    def forward(self, x0, x1, a0, a1, weights):
        size1 = a0.size()[0]
        size2 = a0.size()[1]
        size3 = x0.size()[0]
        size4 = x0.size()[1]
        x_start = torch.zeros(size3, size4).to(self.device)
        a_start = torch.ones(size1, size2).to(self.device)
        states = [[x0, a0], [x1, a1]]
        offset = 0
        for i in range(self._steps):
            x_new = torch.zeros(size3, size4).to(self.device)
            a_new = torch.ones(size1, size2).to(self.device)
            for j, h in enumerate(states):
                x, a = self._ops[offset + j](h[0], h[1], weights[offset + j])
                x_new = x + x_new
                a_new = a * a_new
            offset += len(states)
            states.append([x_new, a_new])
        for j, h in enumerate(states[-self._steps:]):
            x_start = x_start + h[0]
        for j, h in enumerate(states[-self._steps:]):
            a_start = a_start * h[1]
        return x_start, a_start


class Network(nn.Module):
    def __init__(self, nfeat, nclass, nhid, dropout, layers, criterion, steps, multiplier, device):
        super(Network, self).__init__()
        self._nhid = nhid
        self._nfeat = nfeat
        self._nclass = nclass
        self._dropout = dropout
        self._layers = layers
        self._multiplier = multiplier
        self._criterion = criterion
        self._steps = steps
        self.blocks = nn.ModuleList()
        self.device = device
        for i in range(layers):
            block = Block(steps=self._steps, nhid=nhid, nclass=nclass, dropout=dropout, nfeat=nfeat, mulitplier=self._multiplier,
                          device=self.device)
            self.blocks += [block]

        self._initialize_alphas()  # initialize the alpha parameter
        self._classifier = nn.Linear(in_features=nhid, out_features=nclass)
        self._fc = nn.Sequential(
            nn.Linear(in_features=nfeat, out_features=nhid),
            nn.ReLU()
        )
        self.relu = nn.ReLU(True)

    def new(self):
        model_new = Network(self._nfeat, self._nclass, self._nhid, self._dropout, self._layers, self._criterion,
                            self._steps, self._multiplier, self.device)
        model_new = model_new.to(self.device)
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def forward(self, feature, adj):
        feature = self._fc(feature)
        x0 = feature
        x1 = feature
        a0 = adj
        a1 = adj
        for i, Block in enumerate(self.blocks):
            weights = F.softmax(self.alphas_normal, dim=-1)
            x3, a3 = Block(x0, x1, a0, a1, weights)
            x3 = self.relu(x3)
            x0 = x1
            a0 = adj
            x1 = x3
            a1 = adj
            if (i == self._layers - 1):
                x1 = self._classifier(x1)
        return x1,x3

    def _loss(self, idx, target_1, feature, adj):
        logits,raw = self(feature, adj)
        return self._criterion(logits[idx], target_1[idx]) + 1 / cau_cos(raw, target_1, idx)

    def _initialize_alphas(self):
        k = sum(1 for i in range(self._steps) for n in range(2 + i))
        num_ops = len(PRIMITIVES)

        self.alphas_normal = Variable(1e-3 * torch.randn(k, num_ops).to(self.device),
                                      requires_grad=True)
        self._arch_parameters = {
            self.alphas_normal
        }

    def arch_parameters(self):
        return self._arch_parameters

    def genotype(self):  # select 2 best operations
        def _parse(weights):
            gene = []
            n = 2
            start = 0
            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()
                edges = sorted(range(i + 2),
                               key=lambda x: -max(W[x][k] for k in range(len(W[x])) ))[
                        :2]
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                            if k_best is None or W[j][k] > W[j][k_best]:
                                k_best = k
                    gene.append((PRIMITIVES[k_best], j))
                start = end
                n += 1
            return gene

        gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
        concat = range(2 + self._steps - self._multiplier, self._steps + 2)
        genotype = Genotype(
            normal=gene_normal, normal_concat=concat,
        )
        return genotype


class Network_reddit(nn.Module):
    def __init__(self, nfeat, nclass, nhid, dropout, layers, criterion, steps, multiplier, device):
        super(Network_reddit, self).__init__()
        self._nhid = nhid
        self._nfeat = nfeat
        self._nclass = nclass
        self._dropout = dropout
        self._layers = layers
        self._multiplier = multiplier
        self._criterion = criterion
        self._steps = steps
        self.blocks = nn.ModuleList()
        self._fc = nn.Sequential(
            nn.Linear(in_features=nfeat, out_features=nhid),
            nn.ReLU()
        )
        self.convs = nn.ModuleList()
        self.device = device
        for i in range(layers):
            block = Block(steps=4, nhid=nhid, nclass=nclass, dropout=dropout, nfeat=nfeat, mulitplier=self._multiplier,
                          device=self.device)
            self.blocks += [block]
        self._initialize_alphas()
        self._classifier = nn.Linear(in_features=nhid, out_features=nclass)

    def new(self):
        model_new = Network_reddit(self._nfeat, self._nclass, self._nhid, self._dropout, self._layers, self._criterion,
                                   self._steps, self._multiplier, self.device)
        model_new = model_new.to(self.device)
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def forward(self, feature, adj):
        feature = self._fc(feature)
        x0 = feature
        x1 = feature
        a0 = adj
        a1 = adj
        for i, Block in enumerate(self.blocks):
            weights = F.softmax(self.alphas_normal, dim=-1)
            x3, a3 = Block(x0, x1, a0, a1, weights)
            x0 = x1
            a0 = a1
            x1 = x3
            a1 = a3
            if (i == self._layers - 1):
                x1 = self._classifier(x1)
        return x1

    def inference(self, x_all, subgraph_loader, device, weights):
        pbar = tqdm(total=len(subgraph_loader.dataset) * len(self.blocks))
        pbar.set_description('Evaluating')
        xs = []
        for batch in subgraph_loader:
            x1 = x_all[batch.n_id.to(x_all.device)].to(device)
            x = self._fc(x1)
            xs.append(x[:batch.batch_size].cpu())
            pbar.update(batch.batch_size)
        x_all = torch.cat(xs, dim=0).to(device)
        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch:
        for i, conv in enumerate(self.blocks):
            xs = []
            for batch in subgraph_loader:
                adj = to_dense_adj(batch.edge_index).squeeze().to(self.device)
                x = x_all[batch.n_id.to(x_all.device)].to(device)
                x = conv(x, x, adj, adj, weights)
                if i < len(self.blocks) - 1:
                    x = x.relu_()
                xs.append(x[:batch.batch_size].cpu())
                pbar.update(batch.batch_size)
            x_all = torch.cat(xs, dim=0)
        pbar.close()
        return x_all

    def _loss(self, idx, target_1, feature, adj):
        logits = self(feature, adj)
        return self._criterion(logits[idx], target_1[idx]) + 1 / cau_cos(logits, target_1, idx)

    def _initialize_alphas(self):
        k = sum(1 for i in range(self._steps) for n in range(2 + i))
        num_ops = len(PRIMITIVES)

        self.alphas_normal = Variable(1e-3 * torch.randn(k, num_ops).to(self.device),
                                      requires_grad=True)  # 如果require_grad 设置为True则所有依赖它的结点都是True
        self._arch_parameters = {
            self.alphas_normal
        }

    def arch_parameters(self):
        return self._arch_parameters

    def genotype(self):  # 选出最好用的两个操作
        def _parse(weights):
            gene = []
            n = 2
            start = 0
            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()
                edges = sorted(range(i + 2),
                               key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[
                        :2]  # 选出top 2
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if k != PRIMITIVES.index('none'):
                            if k_best is None or W[j][k] > W[j][k_best]:
                                k_best = k
                    gene.append((PRIMITIVES[k_best], j))
                start = end
                n += 1
            return gene

        gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
        concat = range(2 + self._steps - self._multiplier, self._steps + 2)
        genotype = Genotype(
            normal=gene_normal, normal_concat=concat,
        )
        return genotype, self.alphas_normal


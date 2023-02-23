# encoding:utf-8
import torch.utils
from args import *
from architect import Architect
import utils
import numpy as np
from new_measure import new_measure
from torch_geometric.datasets import Planetoid
from new_model_search import *
from torch_geometric.utils import to_dense_adj
import datetime
class NAS_search_trainer:
    def __init__(self ,layers,device,dataset):
        self.nhid = args.nhid
        self.lr = 0.001
        self.weight_decay = args.weight_decay
        self.layers = layers
        self.device = torch.device(device)
        self.dataset = dataset
        dataset = Planetoid(root='../DATA/', name=dataset)
        data = dataset[0]
        features = data.x
        adj_raw = data.edge_index
        adj = to_dense_adj(adj_raw).squeeze()
        labels = data.y
        idx_train, idx_val, idx_test = data.train_mask ,data.val_mask,data.test_mask
        self.nclass = dataset.num_classes
        self.nfeat = dataset.num_features
        self.features = features.to(self.device)
        self.adj = adj.to(self.device)
        self.labels = labels.to(self.device)
        self.idx_train = idx_train.to(self.device)
        self.idx_val = idx_val.to(self.device)
        self.idx_test = idx_test.to(self.device)
        self.mulitplier = 4
        self.steps = 3
        self.mask1 = self.mask1 = torch.ones(self.features.size(0),self.features.size(0)).numpy()
        self.nb_class = (torch.max(data.y) + 1).numpy()
        self.Y_onehot = torch.zeros(data.y.shape[0], self.nb_class).scatter_(1, data.y.unsqueeze(-1), 1).to(self.device)
        nb_each_class_train = torch.sum(self.Y_onehot[data.test_mask], dim=0).to(self.device)
        nb_each_class_inv_train = torch.tensor(np.power(nb_each_class_train.cpu().numpy(), -1).flatten()).to(
            self.device)
        self.nb_each_class_inv_mat_train = torch.diag(nb_each_class_inv_train).to(self.device)
    def train_section(self):
        criterion = nn.CrossEntropyLoss().to(self.device)
        self.model = Network(nfeat=self.nfeat, nclass=self.nclass, nhid=self.nhid, dropout=args.dropout, layers=self.layers, criterion=criterion,
                        steps=self.steps, multiplier=self.mulitplier,device=self.device)
        self.model = self.model.to(self.device)
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay)
        optimizer1 = torch.optim.SGD(
            self.model.parameters(),
            momentum = 0.9,
            lr=self.lr,
            weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, float(50), eta_min=0.001)
        architect = Architect(self.model, args,self.device)
        temp = 0
        for epoch in range(1000):
            lr = scheduler.get_last_lr()[0]
            genotype = self.model.genotype()
            # training
            loss = self.train(self.features, self.adj, self.model, architect, criterion, optimizer, lr)
            # validation
            val_acc,mad_1, gfd_1, f1_macro_1, our_1_1 = self.infer(self.features, self.adj, self.model, criterion)
            scheduler.step()
            
            if our_1_1 >= temp:
                temp = our_1_1
                t = genotype
            final = []
            final.append(our_1_1)
            final.append(val_acc)
            final.append(mad_1)
            final.append(gfd_1)
            final.append(f1_macro_1)
            final.append(t)
            final.append(epoch)
            final.append(loss)
            final = str(final)
            now=datetime.datetime.now().date()
            with open('./'+str(now)+self.dataset+'.txt', 'a') as f:
              f.write(final)
              f.write('\r\n')

    def train(self,features, adj, model, architect, criterion, optimizer, lr):
        model.train()
        architect.step(self.idx_train, self.labels, self.idx_val, self.labels, lr, optimizer, unrolled=False, feature=features, adj=adj)
        optimizer.zero_grad()
        logits,raw = model(features, adj)
        loss = criterion(logits[self.idx_train], self.labels[self.idx_train]) + 1 / cau_cos(raw, self.labels,
                                                                    self.idx_train) 
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        loss.backward(retain_graph=True)
        optimizer.step()

        return loss

    def infer(self,features, adj, model, criterion):
        model.eval()
        logits,raw = model(features, adj)
        prec1 = utils.accuracy(logits[self.idx_val], self.labels[self.idx_val])
        mad_1, gfd_1, f1_macro_1, our_1_1 = new_measure(raw, logits, adj, self.mask1, self.Y_onehot,
                                                        self.nb_each_class_inv_mat_train, self.idx_val,
                                                        self.labels, self.device)
        return prec1,mad_1, gfd_1, f1_macro_1, our_1_1

model = NAS_search_trainer(layers=2,device="cuda:5",dataset='citeseer')
model.train_section()

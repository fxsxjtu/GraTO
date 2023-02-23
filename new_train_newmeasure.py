# encoding:utf-8
from torch_geometric.datasets import Planetoid
from utils import *
import utils
import torch.nn as nn
import torch.utils
from args import *
from model import NetworkCora as Network
from new_measure import *
from torch_geometric.utils import  to_dense_adj


class NAS_trainer:
    def __init__(self, nhid, lr, weight_decay, device, layers, path ,data_name ,derive,alpha):
        self.nhid = nhid
        self.lr = lr
        self.derive = derive
        self.weight_decay = weight_decay
        self.alpha = alpha
        self.layers = layers
        self.data_name = data_name
        self.device = torch.device(device)
        dataset = Planetoid(root='./', name=data_name)
        data = dataset[0]
        adj_raw = data.edge_index
        adj = to_dense_adj(adj_raw).squeeze()
        self.nb_class = (torch.max(data.y) + 1).numpy()
        self.Y_onehot = torch.zeros(data.y.shape[0], self.nb_class).scatter_(1, data.y.unsqueeze(-1), 1).to(self.device)
        nb_each_class_train = torch.sum(self.Y_onehot[data.test_mask], dim=0).to(self.device)
        nb_each_class_inv_train = torch.tensor(np.power(nb_each_class_train.cpu().numpy(), -1).flatten()).to(
            self.device)
        self.nb_each_class_inv_mat_train = torch.diag(nb_each_class_inv_train).to(self.device)
        self.features = data.x.to(self.device)
        self.adj = adj.to(self.device)
        self.labels = data.y.to(self.device)
        self.idx_train = data.train_mask.to(self.device)
        self.idx_val = data.val_mask.to(self.device)
        self.idx_test = data.test_mask.to(self.device)
        self.mask1 = torch.ones(self.features.size(0),self.features.size(0)).numpy()
        self.path = path
    def train_session(self):
        genotypes = eval("genotype.%s" % self.derive)
        self.model = Network(self.features.size(1), self.nb_class, self.nhid, 0.5,self.layers, genotypes,
                             self.device)
        self.model.to(self.device)
        criterion = nn.CrossEntropyLoss().to(self.device)
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay)
        temp_newour = 0
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(100))
        for epoch in range(500):
            ####################################
            train_obj, loss = self.train(self.features, self.adj, self.model, criterion, optimizer)

            scheduler.step()
            val_acc, test_acc, mad, gfd, f1_macro, our_1, our_1_1,raw= self.infer(self.features, self.adj, self.model, criterion)
            if our_1_1 >= temp_newour:
                temp_test_acc_1 = test_acc
                temp_mad_1 = mad
                temp_f1_1 = f1_macro
                temp_our_1 = our_1
                raw_new = raw
        torch.save(raw_new,'./' + self.data_name + str(temp_test_acc_1)+'.pt')
        final = []
        final.append(self.lr)
        final.append(self.weight_decay)
        final.append(self.data_name)
        final.append(self.layers)
        final.append('choose_best_our')
        final.append(temp_test_acc_1)
        final.append(temp_mad_1)
        final.append(temp_f1_1)
        final = str(final)
        with open(self.path, 'a') as f:
            f.write(final)
            f.write('\r\n')

    def train(self, features, adj, model, criterion, optimizer):
        objs = utils.AvgrageMeter()
        top1 = utils.AvgrageMeter()
        # top5 = utils.AvgrageMeter()
        n1 = features[self.idx_test].size(0)
        n2 = features[self.idx_train].size(0)
        model.train()
        optimizer.zero_grad()
        logits, raw = model(features, adj)
        loss = criterion(logits[self.idx_train], self.labels[self.idx_train])  + self.alpha * 1 / cau_cos(raw, self.labels,
                                                                    self.idx_train)
        loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        objs.update(loss.item(), n2)
        return objs.avg, loss 

    def infer(self, features, adj, model, criterion):
        model.eval()
        logits, raw = model(features, adj)
        mad_1, gfd_1, f1_macro_1, our_1_1 = new_measure(raw, logits, adj, self.mask1, self.Y_onehot,
                                                        self.nb_each_class_inv_mat_train, self.idx_val,
                                                        self.labels, self.device)
        mad, gfd, f1_macro, our_1 = new_measure(raw, logits, adj, self.mask1, self.Y_onehot,
                                                self.nb_each_class_inv_mat_train, self.idx_test,
                                                self.labels, self.device)
        prec1 = utils.accuracy(logits[self.idx_val], self.labels[self.idx_val])

        prec2 = accuracy(logits[self.idx_test], self.labels[self.idx_test])

        return prec1, prec2, mad, gfd, f1_macro, our_1, our_1_1,raw




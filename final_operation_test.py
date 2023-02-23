#encoding:utf-8
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch_geometric.nn import GCNConv , GATConv ,SAGEConv,AGNNConv,SGConv
from layer import PairNorm
import random
from torch_geometric.utils import dropout_adj,to_dense_adj,dense_to_sparse
OPS ={
    'none':lambda nfeat,nhid,nclass,dropout:Zero(),
    'GCN_norm' :lambda nfeat,nhid,nclass,dropout:GCN_norm(nfeat,nhid,nclass,dropout),
    'GAT_norm' :lambda nfeat,nhid,nclass,dropout:GAT_norm(nfeat,nhid,nclass,dropout),
    'GraphSAGE_norm' :lambda nfeat,nhid,nclass,dropout:GraphSAGE_norm(nfeat,nhid,nclass,dropout),
     'skip_connect':lambda nfeat,nhid,nclass,dropout:Identity(),
     'AGNN_norm':lambda nfeat,nhid,nclass,dropout:AGNN_norm(nfeat,nhid,nclass,dropout),
     'SGC_norm':lambda nfeat,nhid,nclass,dropout:SGC_norm(nfeat,nhid,nclass,dropout),
     'Drop_attr3':lambda nfeat,nhid,nclass,dropout:Drop_attr3(),
     'Drop_attr4':lambda nfeat,nhid,nclass,dropout:Drop_attr4(),
     'Drop_attr5':lambda nfeat,nhid,nclass,dropout:Drop_attr5(),
     'Drop_Edge':lambda nfeat,nhid,nclass,dropout:Drop_Edge(),
}#���޸ģ���ȷʹ����Щ��
class SGC_norm(nn.Module):
    def __init__(self,nfeat ,nhid , nclass , dropout ,norm_mode='PN-SI', norm_scale=1):
        super().__init__()
        self.gc = SGConv(nhid,nhid)
        self.dropout = dropout
        self.norm = PairNorm(norm_mode,norm_scale)
        self.dropout = dropout
    def forward(self, x ,adj):
        a,b = dense_to_sparse(adj)
        x = F.dropout(x, self.dropout, training=True)
        x = self.gc(x,a)
        x = self.norm(x)
        x = F.relu(x)
        return x ,adj
class AGNN_norm(nn.Module):
    def __init__(self,nfeat,nhid,nclass,dropout,norm_mode='PN-SI', norm_scale=1):
        super().__init__()
        self.gc = AGNNConv(requires_grad=True,add_self_loops=True)
        self.norm = PairNorm(norm_mode,norm_scale)
        self.dropout = dropout
    def forward(self, x ,adj):
        a,b = dense_to_sparse(adj)
        x = F.dropout(x, self.dropout, training=True)
        x = self.gc(x,a)
        x = self.norm(x)
        x = F.relu(x)
        return x ,adj
class GCN_norm(nn.Module):
    def __init__(self,nfeat ,nhid , nclass , dropout ,norm_mode='PN-SI', norm_scale=1):
        super(GCN_norm, self).__init__()
        self.gc = GCNConv(in_channels=nhid,out_channels=nhid,normalize=False,cached=False)
        self.dropout = dropout
        self.norm = PairNorm(norm_mode,norm_scale)
        #self.dropout = nn.Dropout(p=dropout)
    def forward(self, x ,adj):
        a,b = dense_to_sparse(adj)
        x = F.dropout(x, self.dropout ,training=True)
        x = self.gc(x,a,b)
        x = self.norm(x)
        x = F.relu(x)
        return x ,adj

class Identity(nn.Module):

  def __init__(self):
    super(Identity, self).__init__()

  def forward(self, x, adj):
    return x,adj

class Zero(nn.Module):

  def __init__(self):
    super(Zero, self).__init__()

  def forward(self, x,adj):
    return x.mul(0.),adj

class GAT_norm(nn.Module):
    def __init__(self,nfeat ,nhid , nclass , dropout ,norm_mode='PN', norm_scale=1):
        super(GAT_norm, self).__init__()
        self.ga1 = GATConv(in_channels=nhid,out_channels=nhid,heads = 1,dropout = 0.6)
        self.norm = PairNorm(norm_mode,norm_scale)
        self.dropout =dropout
    def forward(self, x, adj):
        a,b = dense_to_sparse(adj)
        x = F.dropout(x, self.dropout ,training=True)
        x = self.ga1(x,a)
        x = self.norm(x)
        x = F.relu(x)
        return x, adj

class GraphSAGE_norm(nn.Module):
    def __init__(self,nfeat ,nhid , nclass , dropout=0.6 ,norm_mode='PN-SI', norm_scale=1):
        super(GraphSAGE_norm, self).__init__()
        self.gs = SAGEConv(in_channels=nhid,out_channels=nhid)
        self.norm = PairNorm(norm_mode, norm_scale)
        self.dropout = dropout

    def forward(self, x, adj):
        a,b = dense_to_sparse(adj)
        x = F.dropout(x, self.dropout ,training=True)
        x = self.gs(x, a)
        x = self.norm(x)
        x = F.relu(x)
        return x, adj

class Drop_Edge(nn.Module):
    def __init__(self):
        super(Drop_Edge, self).__init__()
    def forward(self,x,adj):
        m,n = dense_to_sparse(adj)
        m,n = dropout_adj(edge_index=m, edge_attr=n, p=0.3)

        t = to_dense_adj(edge_index=m,edge_attr=n,max_num_nodes=adj.size(0)).squeeze()
        return x ,t


class Drop_attr1(nn.Module):#switch
    def __init__(self):
        super(Drop_attr1, self).__init__()
    def forward(self,x,adj):
        row = x.size()[0]
        i = random.randint(0, row-1)
        j = random.randint(0, row-1)
        t = x[i]
        x[i] = x[j]
        x[j] = t
        return x ,adj

class Drop_attr2(nn.Module):#switch
    def __init__(self):
        super(Drop_attr2, self).__init__()
    def forward(self,x,adj):
        col = x.size()[1]
        i = random.randint(0, col-1)
        j = random.randint(0, col-1)
        t = x[:,i]
        x[:,i] = x[:,j]
        x[:,j] = t
        return x ,adj


class Drop_attr3(nn.Module):
    def __init__(self):
        super(Drop_attr3, self).__init__()
    def forward(self,x,adj):
        x = x + 0
        col = x.size()[1]
        i = random.randint(0, col-1)
        x[:,i].fill_(0)
        return x ,adj



class Drop_attr4(nn.Module):
    def __init__(self):
        super(Drop_attr4, self).__init__()
    def forward(self,x,adj):
        x = x + 0
        row = x.size()[0]
        i = random.randint(0, row - 1)
        x[i].fill_(0)
        return x ,adj

class Drop_attr5(nn.Module):
    def __init__(self):
        super(Drop_attr5, self).__init__()
    def forward(self,x,adj):
        x = torch.nn.functional.dropout(x, p=0.6,training=True, inplace=False)
        return x,adj


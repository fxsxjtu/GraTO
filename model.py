import torch
import torch.nn as nn
from final_operation_test import *
from torch.autograd import Variable
from layer import PairNorm
from utils import drop_path
from genotype import PRIMITIVES
from architect import Architect
from genotype import Genotype

class Block(nn.Module):
  def __init__(self, nhid, nclass ,dropout,nfeat,genotype, device):#, nfeats_prev_prev,adj_prev, adj_prev_prev):
    super(Block, self).__init__()
    # self.preprocess0 = #���������һ����в���
    self.device = device
    # self.preprocess1 =
    op_names ,indices = zip(*genotype.normal)
    concat = genotype.normal_concat
    self._compile(nfeat, nhid, nclass, dropout, op_names, indices, concat)

  def _compile(self,nfeat, nhid, nclass, dropout, op_names, indices, concat):
    assert len(op_names) == len(indices)
    self._steps = len(op_names)//2
    self._concat = concat
    self.multiplier = len(concat)
    self._ops = nn.ModuleList()
    for name, index in zip(op_names, indices):
      op = OPS[name](nfeat, nhid, nclass ,dropout)
      self._ops.append(op)
    self._indices = indices

  def forward(self, x0 ,x1 , a0 ,a1):#x����������㣬��aֻӰ�����
    size1 = a0.size()[0]
    size2 = a0.size()[1]
    size3 = x0.size()[0]
    size4 = x0.size()[1]
    x_start = torch.zeros(size3, size4).to(self.device)
    a_start = torch.ones(size1, size2).to(self.device)
    x_start = x_start
    a_start = a_start
    states = [[x0,a0],[x1,a1]]
    for i in range(self._steps):
      h1 = states[self._indices[2*i]][0]
      t1 = states[self._indices[2*i]][1]
      h2 = states[self._indices[2 * i+1]][0]
      t2 = states[self._indices[2 * i+1]][1]
      op1 = self._ops[2 * i]
      op2 = self._ops[2 * i + 1]
      h1,t1 = op1(h1,t1)
      h2 ,t2= op2(h2,t2)
      s = h1 + h2
      t = t1 * t2
      #print("#################################################################################")
      #print(t.shape)
      states.append([s,t])
    for j, h in enumerate(states[-4:]):
      # print(x_start.device)
      # print(h[0].shape)
      x_start = x_start + h[0]
    for j, h in enumerate(states[-4:]):  # �����е�A����һ�£�����ٷ��ظ�ֵ
      a_start = a_start * h[1]
    return x_start, a_start

class NetworkCora(nn.Module):
  def __init__(self, nfeat, nclass, nhid, dropout, layers, genotype, device,norm_mode='PN', norm_scale=1):
    super(NetworkCora, self).__init__()
    self._nhid = nhid
    self._nfeat = nfeat
    self._nclass = nclass
    self._dropout = dropout
    self._layers = layers
    self.blocks = nn.ModuleList()
    self.device = device
    for i in range(layers):
      block = Block(nhid, nclass ,dropout,nfeat,genotype, self.device)
      self.blocks += [block]

    self._fc = nn.Sequential(
        nn.Linear(in_features=nfeat,out_features=nhid),
        nn.ReLU()
    )
    self.norm = PairNorm(norm_mode, norm_scale)

    self._classifier = nn.Linear(in_features=nhid,out_features=nclass)
    self.relu = nn.ReLU(True)
  def forward(self, feature ,adj):
    feature=self._fc(feature)
    x0 = feature
    x1 = feature
    a0 = adj
    a1 = adj
    for i, Block in enumerate(self.blocks):
      x3 , a3 = Block(x0,x1,a0,a1)
      #x3 = self.norm(x3)
      x3 = self.relu(x3)
      x0 = x1
      a0 = a1
      x1 = x3
      a1 = a3
      if(i == self._layers - 1):
        x1 = self._classifier(x1)
    return x1,x3

import os
import numpy as np
import torch
import shutil
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
from sklearn.metrics import f1_score
from torch_geometric.utils import dropout_adj,to_dense_adj,dense_to_sparse
class AvgrageMeter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt


# def accuracy(output, target, topk=(1,)):
#   maxk = max(topk)
#   batch_size = target.size(0)
#
#   _, pred = output.topk(maxk, 1, True, True)
#   pred = pred.t()
#   correct = pred.eq(target.view(1, -1).expand_as(pred))
#
#   res = []
#   for k in topk:
#     correct_k = correct[:k].view(-1).float().sum(0)
#     res.append(correct_k.mul_(100.0/batch_size))
#   return res

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    #print(torch.max(preds))
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

class Cutout(object):#数据增强代码，生成一个边长为length的mask
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)#clip 防止越界
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms_cifar10(args):
  CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
  CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

  train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
  ])
  if args.cutout:
    train_transform.transforms.append(Cutout(args.cutout_length))

  valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
  return train_transform, valid_transform


def count_parameters_in_MB(model):
  return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6


def save_checkpoint(state, is_best, save):
  filename = os.path.join(save, 'checkpoint.pth.tar')
  torch.save(state, filename)
  if is_best:
    best_filename = os.path.join(save, 'model_best.pth.tar')
    shutil.copyfile(filename, best_filename)


def save(model, model_path):
  torch.save(model.state_dict(), model_path)


def load(model, model_path):
  model.load_state_dict(torch.load(model_path))


def drop_path(x, drop_prob):
  if drop_prob > 0.:
    keep_prob = 1.-drop_prob
    mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
    x.div_(keep_prob)
    x.mul_(mask)
  return x


def create_exp_dir(path, scripts_to_save=None):
  if not os.path.exists(path):
    os.mkdir(path)
  print('Experiment dir : {}'.format(path))

  if scripts_to_save is not None:
    os.mkdir(os.path.join(path, 'scripts'))
    for script in scripts_to_save:
      dst_file = os.path.join(path, 'scripts', os.path.basename(script))
      shutil.copyfile(script, dst_file)
      
def cau_cos(logits ,label ,idx):
    batch_size = 20
    sample_idxs = np.random.choice(range(len(idx)), size=batch_size)
    batch_xs = []
    batch_ys = []

    # val_sample_idxs = np.random.choice(range(len(idx_val)), size=batch_size)
    # val_batch_xs = []
    # val_batch_ys = []
    cos = nn.CosineSimilarity(dim=0, eps=1e-6)
    for j in range(batch_size):
        train_id = sample_idxs[j]
        batch_xs.append(logits[train_id])
        batch_ys.append(label[train_id])
        # val_id = val_sample_idxs[j]
        # val_batch_xs.append(logits[val_id])
        # val_batch_ys.append(label[val_id])
    total = 0
    count = 0
    for i in range(batch_size):
        for j in range(i, batch_size):
            if (batch_ys[i] != batch_ys[j]):
                temp =1- cos(batch_xs[i], batch_xs[j])
                total = total+temp
                count = count+1
    return total/count

def get_f1score(logits ,labels):
    preds = logits.cpu().detach().max(1)[1].type_as(labels)
    out1 = f1_score(labels,preds,average='macro')
    out2 = f1_score(labels,preds,average='micro')
    return out1,out2
    
           
    
def get_y_hot_nb_class_adj(data,device):
    adj_raw = data.edge_index
    adj = to_dense_adj(adj_raw).squeeze()
    nb_class = (torch.max(data.y) + 1).cpu().numpy()
    Y_onehot = torch.zeros(data.y.shape[0], nb_class).scatter_(1, data.y.unsqueeze(-1), 1).to(device)
    nb_each_class_train = torch.sum(Y_onehot[data.test_mask], dim=0).to(device)
    nb_each_class_inv_train = torch.tensor(np.power(nb_each_class_train.cpu().numpy(), -1).flatten()).to(
    device)
    nb_each_class_inv_mat_train = torch.diag(nb_each_class_inv_train).to(device) 
    return Y_onehot,nb_each_class_inv_mat_train,adj
    
def cau_cos_reddit(logits ,label ,idx):
    batch_size = 10
    sample_idxs = np.random.choice(range(idx), size=batch_size)
    batch_xs = []
    batch_ys = []

    # val_sample_idxs = np.random.choice(range(len(idx_val)), size=batch_size)
    # val_batch_xs = []
    # val_batch_ys = []
    cos = nn.CosineSimilarity(dim=0, eps=1e-6)
    for j in range(batch_size):
        train_id = sample_idxs[j]
        batch_xs.append(logits[train_id])
        batch_ys.append(label[train_id])
        # val_id = val_sample_idxs[j]
        # val_batch_xs.append(logits[val_id])
        # val_batch_ys.append(label[val_id])
    total = 0
    for i in range(batch_size):
        for j in range(i, batch_size):
            if (batch_ys[i] != batch_ys[j]):
                temp =1- cos(batch_xs[i], batch_xs[j])
                total = total+temp
    return total/batch_size
def cau_cos_ppi(logits ,label ,idx):
    batch_size = 20
    #print(logits.size())
    #print(label.size())
    sample_idxs = np.random.choice(range(121), size=batch_size)
    batch = []
    total = 0
    cos = nn.CosineSimilarity(dim=0, eps=1e-6)
    all = cos(logits,label)
    for j in range(batch_size):
        train_id = sample_idxs[j]
        total = total + all[train_id].item()
    print(total)
    
    return total
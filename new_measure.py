from sklearn.metrics import f1_score,roc_auc_score
from Mad_measure import *
from LDA_loss import *

def new_measure(other ,output, adj ,mask ,Y_onehot ,nb_each_class_inv ,index,labels,device):
    logits_numpy = other.cpu().detach().numpy()
    MAD = mad_value(logits_numpy,mask)
    temp = torch.spmm(adj,output)
    GFD = LDA_loss(temp[index], Y_onehot[index],nb_each_class_inv)
    F1_macro = get_f1score(output,labels ,index,device)
    ours2 = 2/(1/MAD + 1/F1_macro)
    F1_macro = round(F1_macro, 4)
    ours2 = round(ours2, 4)
    return MAD,GFD,F1_macro,ours2  #GFD is a useless parameter in the following code, so don't mind


def get_f1score(logits ,labels,index,device):
    preds = logits.max(1)[1].detach().cpu().type_as(labels)[index].cpu()
    out1 = f1_score(labels[index].cpu(),preds,average='macro')
    return out1
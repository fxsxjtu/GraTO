from normalization import normalization
# from plots import plot_feature
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from normalization import *

def norm_expand_tensor(feature):
    assert len(feature.shape) == 2
    mean = feature.mean(dim=0, keepdim=True)
    var = feature.std(dim=0, keepdim=True)
    return (feature - mean) / (var + 1e-6)

def LDA_loss_of_a_pair(mu1, mu2, sigma1_vec, sigma2_vec): #need to double check
    '''
    mu1, mu2 are two mean vectors of dim d for class 1 and 2
    sigma1, sigma2 are two variance matrix of dim nd*d for class 1 and 2
    '''
    sigma_sum = sigma1_vec + sigma2_vec + 1e-6 # d * 1
    w = (1/sigma_sum).mul(mu1-mu2) # d * 1
    J = w.mul(mu1-mu2).sum()
    #print("mu=", mu1 , mu2, "sigma=", sigma1_vec, sigma2_vec, "w=", w)
    return J

def LDA_loss(H, Y_onehot, nb_each_class_inv_mat, norm_or_not=True):
    '''
    H is representation matrix of dim n * d
    Y_onehot is a one-hot matrix of dim n * c, c is number of class
    nb_each_class_inv_mat is a diagonal matrix of dim c * c
    this loss encourage node in different class to be as linear seperable as possible
    '''
    result = 0
    weight_sum = 0
    if norm_or_not:
        #         do expand_norm won't effect LDA_loss
        #         print(norm_or_not)
        H = norm_expand_tensor(H)

    #     step1: get shape
    nb_nodes = Y_onehot.shape[0]
    nb_class = Y_onehot.shape[1]

    #     step2: get mean_mat, each column is a mean vector for a class
    H_T = torch.transpose(H, 0, 1)  # transpose of matrix H
    sum_mat = torch.mm(H_T, Y_onehot)  # d * c
    mean_mat = torch.mm(sum_mat, nb_each_class_inv_mat)  # d * c

    #     step3: get var_mat, each colums is a variance vector for a class
    '''
    var(X) = mean(X^2) - mean(X)^2 
    '''
    H2 = H.mul(H)  # each item in H2 is the square of corresponding item in H
    H2_T = torch.transpose(H2, 0, 1)  # transpose of matrix H2
    sum_mat2 = torch.mm(H2_T, Y_onehot)  # d * c
    mean_mat2 = torch.mm(sum_mat2, nb_each_class_inv_mat)  # d * c
    var_mat = mean_mat2 - mean_mat.mul(mean_mat)  # d * c
    var_mat = torch.relu(var_mat)

    #     step4: for each pair, get weight and score
    for i in range(nb_class):
        for j in range(i + 1, nb_class):
            weight = 1 / nb_each_class_inv_mat[i][i].cpu().numpy() + 1 / nb_each_class_inv_mat[j][j].cpu().numpy()
            score = LDA_loss_of_a_pair(mean_mat[:, i], mean_mat[:, j], var_mat[:, i], var_mat[:, j])
            weight_sum = weight_sum + weight
            result = result + weight * score

    result = result / weight_sum

    return result
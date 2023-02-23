import scipy.sparse as sp
import torch
import torch.nn.functional as F
import numpy as np

def normalize_row(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.#turn inf into 0
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def normalize_sym(adj):
    """Symmetrically normalize adjacency matrix."""
#     adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    mx = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
#     mx = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return mx

def normalize_col(mx):
    """Column-normalize sparse matrix"""
#     mx here corresponds to adj matrix
    colsum = np.array(mx.sum(-1))
    c_inv = np.power(colsum, -1).flatten()
    c_inv[np.isinf(c_inv)] = 0.
    c_mat_inv = sp.diags(c_inv)
    mx = mx.dot(c_mat_inv)
    return mx

def normalization(A, norm_method, num_power):
    '''
    input dense matrix A
    norm_method: row, col, sym
    num_power: do feature propagation how many times
    '''
    
    A = sp.coo_matrix(A)
    
    if norm_method == 'row':
        # print("row normalization, propagate ", num_power, " times.")
        A = normalize_row(A)
    elif norm_method == 'col':
        # print("column normalization, propagate ", num_power, " times.")
        A = normalize_col(A)
    else:
        # print("symmetric normalization, propagate ", num_power, " times.")
        A = normalize_sym(A)
    
    A = sp.csr_matrix(A)
    res = A
#     print(res)
    while (num_power>1):
#         print(num_power)
        res = res * A
        num_power = num_power - 1
#         print(res)
    return res

def get_list_of_normalized_adjs(A,degree):
    '''
    for AGNN
    '''
    norm_method = ['row', 'col', 'sym']
    adjs = []
    for method in norm_method:
        for i in range(degree):
#             print(method,i+1)
            adj_temp = normalization(A, method, i+1)
            adj_temp = sparse_mx_to_torch_sparse_tensor(adj_temp)
            adjs.append(adj_temp)
    return adjs


def sgc_precompute(features, adj, degree):
    for i in range(degree):
        features = torch.spmm(adj, features)
    return features


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape).cuda()


def get_adj_feats(adj, feats, model_opt, degree,weights):
    '''
    input adjacency, feature tensor
    output required adj and feats for model_opt
    '''
    A = adj.numpy()
    if model_opt == 'GCN':
        A = normalization(A = A, norm_method = 'sym', num_power = 1)
        adj = sparse_mx_to_torch_sparse_tensor(A)
        print("for GCN, return sym_norm(A) and raw feats")
        return adj, feats
    elif model_opt == 'SGC':
        identity = sparse_mx_to_torch_sparse_tensor(sp.eye(len(A)))
        A = normalization(A = A, norm_method = 'sym', num_power = 1)
        adj = sparse_mx_to_torch_sparse_tensor(A)
        feats = sgc_precompute(features = feats, adj = adj, degree = degree)
        print("for SGC, return identity matrix and propagated feats")
        return identity, feats
    elif model_opt == 'GFNN':
        identity = sparse_mx_to_torch_sparse_tensor(sp.eye(len(A)))
        A = normalization(A = A, norm_method = 'sym', num_power = 1)
        adj = sparse_mx_to_torch_sparse_tensor(A)
        feats = sgc_precompute(features = feats, adj = adj, degree = degree)
        print("for GFNN, return identity matrix and propagated feats")
        return identity, feats
    elif model_opt == 'GFN':
        d = np.array(A.sum(1))
        d = np.reshape(d,(-1,len(d))).T
        adj = sparse_mx_to_torch_sparse_tensor(sp.eye(len(A)))
        A = normalization(A = A, norm_method = 'sym', num_power = 1)
        feat_spar = sp.coo_matrix(feats.numpy())
        
        gfn_list = []
        gfn_list.append(d)
        gfn_list.append(feat_spar)
        adj_temp = feat_spar
        for i in range(degree):
            adj_temp = A * adj_temp
            gfn_list.append(adj_temp)
        
        feats = sp.hstack(gfn_list)
        feats = sparse_mx_to_torch_sparse_tensor(feats).to_dense()
        #feats = torch.FloatTensor(np.array(feats.todense())).float()
        print("for GFN, return identity matrix and the concatenation of propagated feats")
        return adj, feats
    elif model_opt == 'PreCompute_AFGNN':
        adj = get_list_of_normalized_adjs(A,degree)
        identity = (sp.eye(len(A)))
        adj_sum= weights[0] * torch.eye(adj[0].shape[1])
        for i in range(len(adj)):
            adj_sum = adj_sum + weights[i+1] * adj[i]
#         A = normalization(A = A, norm_method = norm_method, num_power = 1)
#         adj_sum = sparse_mx_to_torch_sparse_tensor(adj_sum)
        feats = torch.spmm(adj_sum, feats)
#         feats = sgc_precompute(features = feats, adj = adj, degree = degree)
        identity = sparse_mx_to_torch_sparse_tensor(identity)
        print("for Precompute, return identity matrix and propagated feats")
    
        return identity, feats
    elif model_opt == 'GIN':
        A = sp.coo_matrix(A)
        adj = sparse_mx_to_torch_sparse_tensor(A)
        print("for GIN, return raw adj matrix and raw feats matrix")
        return adj, feats
    elif model_opt == 'AGNN':
        adj = get_list_of_normalized_adjs(A,degree)
        return adj, feats
    else:
        raise NotImplementedError('model:{} is not implemented!'.format(model_opt))
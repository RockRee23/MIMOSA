import scipy.sparse as sp
import numpy as np
import torch
import torch.nn as nn

def cal_bpr_loss(preds):#(batch_size, item_num)#[item_num=1 pos+k neg]
    if preds.shape[1] > 2:
        neg = preds[:, 1:]
        pos = preds[:, 0].unsqueeze(1).expand_as(neg)
    elif preds.shape[1] == 2:
        neg = preds[:, 1].unsqueeze(1)
        pos = preds[:, 0].unsqueeze(1)
    loss = - torch.mean(torch.log(torch.sigmoid(pos - neg)+1e-24))
    return loss

def laplace_transform(graph):
    rowsum_sqrt = sp.diags(1/(np.sqrt(graph.sum(axis=1).A.ravel()) + 1e-8))
    colsum_sqrt = sp.diags(1/(np.sqrt(graph.sum(axis=0).A.ravel()) + 1e-8))
    graph = rowsum_sqrt @ graph @ colsum_sqrt#@表示矩阵乘法
    return graph

def to_tensor(graph):
    graph = graph.tocoo()
    values = graph.data
    indices = np.vstack((graph.row, graph.col))
    graph = torch.sparse.FloatTensor(torch.LongTensor(indices), torch.FloatTensor(values), torch.Size(graph.shape)).cuda()
    return graph

def np_edge_dropout(values, dropout_ratio):
    mask = np.random.choice([0, 1], size=(len(values),), p=[dropout_ratio, 1-dropout_ratio])
    values = mask * values
    return values

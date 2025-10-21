import os
import random
import numpy as np
import scipy.sparse as sp 
from collections import defaultdict
import torch


def get_bi(path, num_b, num_i):
    with open(os.path.join(path, 'bundle_item.txt'), 'r') as f:
        bi_pairs = list(map(lambda s: tuple(int(i) for i in s[:-1].split('\t')), f.readlines()))
        
    indice = np.array(bi_pairs, dtype=np.int32)
    values = np.ones(len(bi_pairs), dtype=np.float32)
    bi_graph = sp.coo_matrix(
        (values, (indice[:, 0], indice[:, 1])), shape=(num_b, num_i)).tocsr()

    return bi_pairs, bi_graph


def get_ui(path, num_u, num_i):
    with open(os.path.join(path, 'user_item.txt'), 'r') as f:
        ui_pairs = list(map(lambda s: tuple(int(i) for i in s[:-1].split('\t')), f.readlines()))
        
    indice = np.array(ui_pairs, dtype=np.int32)
    values = np.ones(len(ui_pairs), dtype=np.float32)
    ui_graph = sp.coo_matrix( 
        (values, (indice[:, 0], indice[:, 1])), shape=(num_u, num_i)).tocsr()

    return ui_pairs, ui_graph

def get_ub(path, task, num_u, num_b):
    with open(os.path.join(path, 'user_bundle_{}.txt'.format(task)), 'r') as f:
        ub_pairs = list(map(lambda s: tuple(int(i) for i in s[:-1].split('\t')), f.readlines()))

    indice = np.array(ub_pairs, dtype=np.int32)
    values = np.ones(len(ub_pairs), dtype=np.float32)
    ub_graph = sp.coo_matrix(
        (values, (indice[:, 0], indice[:, 1])), shape=(num_u, num_b)).tocsr()

    return ub_pairs, ub_graph

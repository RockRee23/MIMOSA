import os
os.environ['CUDA_VISIBLE_DEVICES']='1'

from model.mimosa_model import MIMOSA
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import argparse
import math
import csv
from time import time
from utils.evalutils import evaluation
from dataload.mimosa_dataset import BundleDataset
from utils.trainutils import training

model_name='mimosa'

model_path='../save_model/{}/{}_{}.pt'

def set_seed(seed): 
    torch.manual_seed(seed) 
    torch.cuda.manual_seed(seed) 
    np.random.seed(seed) 
    random.seed(seed)
    
def create_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=2023)
    parser.add_argument('--data', default='Youshu')#Youshu, NetEase
    parser.add_argument('--emb_dim', default=64)
    parser.add_argument('--batch_size_train', default=2048)
    parser.add_argument('--batch_size_test', default=4096)
    parser.add_argument('--lr', default=5e-3)
    parser.add_argument('--num_epoch', default=100)
    parser.add_argument('--l2_reg', default=1e-5)
    parser.add_argument('--c0_reg', default=0.03)
    parser.add_argument('--c1_reg', default=0.02)
    parser.add_argument('--c2_reg', default=0.02)
    parser.add_argument('--c_temp', default=0.2)#[0.05, 0.1,0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
    parser.add_argument("--UB_ratio", default=0.2)
    parser.add_argument("--UI_ratio", default=0.1)
    parser.add_argument("--BI_ratio", default=0.1)
    parser.add_argument("--fusion_weights", default={'modal_weight': [0.35, 0.15, 0.5], 'UB_layer': [0.7, 0.2, 0.1], 'UI_layer': [0.6, 0.2, 0.2], 'BI_layer': [0.6, 0.1, 0.3]})
    parser.add_argument('--aug_type', default='Noise')#['ED', 'MD', 'OP', 'Noise']
    parser.add_argument('--num_layers', default=2)
    parser.add_argument('--topks', default=[20,40])
    parser.add_argument('--num_neg', default=10)
    parser.add_argument('--num_neigh', default=2)
    return parser
            
def main():
    parser = create_argument_parser()
    args = parser.parse_args(args=[])
    set_seed(args.seed) 
    model_save = model_path.format(args.data, model_name, args.seed)
    
    dataset = BundleDataset(args.data, args.num_neg, args.num_neigh)
    args.num_usr, args.num_bnd, args.num_itm = dataset.num_usr, dataset.num_bnd, dataset.num_itm
    model = MIMOSA(dataset.graphs, args)
    model.cuda()

    train_bnd_loader = dataset.get_train_bnd_loader(args.batch_size_train)
    val_bnd_loader = dataset.get_val_bnd_loader(args.batch_size_test)
    test_bnd_loader = dataset.get_test_bnd_loader(args.batch_size_test)
    
    optimizer = optim.Adam(model.parameters(), args.lr, weight_decay=args.l2_reg)
    best_recall = - np.inf
    for epoch in range(args.num_epoch):
        time1=time()
        loss = training(model, optimizer, train_bnd_loader, 'bnd')
        time2=time()
        print('Iteration %d, loss is [%.4f], time:%.4f' % (epoch, loss, time2-time1))      
        
        metrics = evaluation(model, val_bnd_loader, args.topks)
        if best_recall < metrics[20]['recall']:
            best_recall =  metrics[20]['recall']
            best_epoch = epoch
            with open(model_save, 'wb') as f:
                torch.save(model, f)
           
    with open(model_save, 'rb') as f:
        model = torch.load(f, map_location='cuda')
        model.cuda()
        metrics = evaluation(model, test_bnd_loader, args.topks)
        m_str = 'Test: '
        for topk in args.topks:
            recall, ndcg = metrics[topk]['recall'], metrics[topk]['ndcg']
            m_str = m_str + 'Rec@{K}:%.4f,NDCG@{K}:%.4f '.format(K=topk)%(recall, ndcg)
        print (m_str)   

            
        
if __name__=='__main__':
    main()

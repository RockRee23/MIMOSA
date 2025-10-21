#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
from time import time
from model.modelutils import cal_bpr_loss, laplace_transform, to_tensor, np_edge_dropout
class MIMOSA(nn.Module):
    def __init__(self, raw_graph, args):
        super(MIMOSA, self).__init__()
        self.usrEmbed = nn.Parameter(torch.FloatTensor(args.num_usr, args.emb_dim))
        self.usrEmbed.cuda()
        nn.init.xavier_normal_(self.usrEmbed)
        self.itmEmbed = nn.Parameter(torch.FloatTensor(args.num_itm, args.emb_dim))
        self.itmEmbed.cuda()
        nn.init.xavier_normal_(self.itmEmbed)
        self.bndEmbed = nn.Parameter(torch.FloatTensor(args.num_bnd, args.emb_dim))
        self.bndEmbed.cuda()
        nn.init.xavier_normal_(self.bndEmbed)
        
        self.num_layers = args.num_layers
        self.c_temp = args.c_temp
        self.c0_reg = args.c0_reg
        self.c1_reg = args.c1_reg
        self.c2_reg = args.c2_reg
        self.num_neigh = args.num_neigh
        self.aug_type = args.aug_type
        self.UB_ratio = args.UB_ratio
        self.UI_ratio = args.UI_ratio
        self.BI_ratio = args.BI_ratio
        

        self.fusion_weights = args.fusion_weights

        self.init_fusion_weights()

        assert isinstance(raw_graph, list)
        self.ub_graph, self.ui_graph, self.bi_graph = raw_graph

        # generate the graph without any dropouts for testing
        self.UB_prop_graph_ori = self.get_propagation_graph(self.ub_graph)

        self.UI_prop_graph_ori = self.get_propagation_graph(self.ui_graph)
        self.UI_aggr_graph_ori = self.get_aggregation_graph(self.ui_graph)

        self.BI_prop_graph_ori = self.get_propagation_graph(self.bi_graph)
        self.BI_aggr_graph_ori = self.get_aggregation_graph(self.bi_graph)

        # generate the graph with the configured dropouts for training, if aug_type is OP or MD, the following graphs with be identical with the aboves
        self.UB_prop_graph_rnd = self.get_propagation_graph(self.ub_graph, self.UB_ratio)

        self.UI_prop_graph_rnd = self.get_propagation_graph(self.ui_graph, self.UI_ratio)
        self.UI_aggr_graph_rnd = self.get_aggregation_graph(self.ui_graph, self.UI_ratio)

        self.BI_prop_graph_rnd = self.get_propagation_graph(self.bi_graph, self.BI_ratio)
        self.BI_aggr_graph_rnd = self.get_aggregation_graph(self.bi_graph, self.BI_ratio)

        if args.aug_type == 'MD':
            self.init_md_dropouts()
        elif args.aug_type == "Noise":
            self.init_noise_eps()
    
    def init_md_dropouts(self):
        self.UB_dropout = nn.Dropout(self.UB_ratio, True).cuda()
        self.UI_dropout = nn.Dropout(self.UI_ratio, True).cuda()
        self.BI_dropout = nn.Dropout(self.BI_ratio, True).cuda()
        self.mess_dropout_dict = {
            "UB": self.UB_dropout,
            "UI": self.UI_dropout,
            "BI": self.BI_dropout
        }


    def init_noise_eps(self):
        self.UB_eps = self.UB_ratio
        self.UI_eps = self.UI_ratio
        self.BI_eps = self.BI_ratio
        self.eps_dict = {
            "UB": self.UB_eps,
            "UI": self.UI_eps,
            "BI": self.BI_eps
        }


    def init_fusion_weights(self):
        assert (len(self.fusion_weights['modal_weight']) == 3), \
            "The number of modal fusion weights does not correspond to the number of graphs"

        assert (len(self.fusion_weights['UB_layer']) == self.num_layers + 1) and\
               (len(self.fusion_weights['UI_layer']) == self.num_layers + 1) and \
               (len(self.fusion_weights['BI_layer']) == self.num_layers + 1),\
            "The number of layer fusion weights does not correspond to number of layers"

        modal_coefs = torch.FloatTensor(self.fusion_weights['modal_weight'])
        UB_layer_coefs = torch.FloatTensor(self.fusion_weights['UB_layer'])
        UI_layer_coefs = torch.FloatTensor(self.fusion_weights['UI_layer'])
        BI_layer_coefs = torch.FloatTensor(self.fusion_weights['BI_layer'])

        self.modal_coefs = modal_coefs.unsqueeze(-1).unsqueeze(-1).cuda()

        self.UB_layer_coefs = UB_layer_coefs.unsqueeze(0).unsqueeze(-1).cuda()
        self.UI_layer_coefs = UI_layer_coefs.unsqueeze(0).unsqueeze(-1).cuda()
        self.BI_layer_coefs = BI_layer_coefs.unsqueeze(0).unsqueeze(-1).cuda()
    
    def get_propagation_graph(self, bipartite_graph, modif_ratio=0):
        propagation_graph = sp.bmat([[sp.csr_matrix((bipartite_graph.shape[0], bipartite_graph.shape[0])), bipartite_graph], [bipartite_graph.T, sp.csr_matrix((bipartite_graph.shape[1], bipartite_graph.shape[1]))]])

        if modif_ratio != 0:
            if self.aug_type == "ED":
                graph = propagation_graph.tocoo()
                values = np_edge_dropout(graph.data, modif_ratio)
                propagation_graph = sp.coo_matrix((values, (graph.row, graph.col)), shape=graph.shape).tocsr()
        norm_graph = to_tensor(laplace_transform(propagation_graph))
        norm_graph.cuda()
        return norm_graph
    

    def get_aggregation_graph(self, bipartite_graph, modif_ratio=0):
        if modif_ratio != 0:
            if self.aug_type == "ED":
                graph = bipartite_graph.tocoo()
                values = np_edge_dropout(graph.data, modif_ratio)
                bipartite_graph = sp.coo_matrix((values, (graph.row, graph.col)), shape=graph.shape).tocsr()

        bundle_size = bipartite_graph.sum(axis=1) + 1e-8
        bipartite_graph = sp.diags(1/bundle_size.A.ravel()) @ bipartite_graph
        norm_graph = to_tensor(bipartite_graph)
        norm_graph.cuda()
        return norm_graph


    def propagate(self, graph, A_feature, B_feature, graph_type, layer_coef, test):
        features = torch.cat((A_feature, B_feature), 0)
        all_features = [features]

        for i in range(self.num_layers):
            features = torch.spmm(graph, features)
            if self.aug_type == "MD" and not test:
                mess_dropout = self.mess_dropout_dict[graph_type]
                features = mess_dropout(features)
            elif self.aug_type == "Noise" and not test:
                random_noise = torch.rand_like(features).cuda()
                eps = self.eps_dict[graph_type]
                features += torch.sign(features) * F.normalize(random_noise, dim=-1) * eps

            all_features.append(F.normalize(features, p=2, dim=1))

        all_features = torch.stack(all_features, 1) * layer_coef
        all_features = torch.sum(all_features, dim=1)
        A_feature, B_feature = torch.split(all_features, (A_feature.shape[0], B_feature.shape[0]), 0)

        return A_feature, B_feature


    def aggregate(self, agg_graph, node_feature, graph_type, test):
        aggregated_feature = torch.matmul(agg_graph, node_feature)

        # simple embedding dropout on bundle embeddings
        if self.aug_type == "MD" and not test:
            mess_dropout = self.mess_dropout_dict[graph_type]
            aggregated_feature = mess_dropout(aggregated_feature)
        elif self.aug_type == "Noise" and not test:
            random_noise = torch.rand_like(aggregated_feature).cuda()
            eps = self.eps_dict[graph_type]
            aggregated_feature += torch.sign(aggregated_feature) * F.normalize(random_noise, dim=-1) * eps

        return aggregated_feature


    def fuse_usr_bnd_feat(self, usr_feat, bnd_feat):
        usr_feat = torch.stack(usr_feat, dim=0)
        bnd_feat = torch.stack(bnd_feat, dim=0)

        # Modal aggregation
        usr_rep = torch.sum(usr_feat * self.modal_coefs, dim=0)
        bnd_rep = torch.sum(bnd_feat * self.modal_coefs, dim=0)

        return usr_rep, bnd_rep


    def get_multi_modal_representations(self, test=False):
        #  =============================  UB graph propagation  =============================
        if test:
            UB_usrGEmb, UB_bndGEmb = self.propagate(self.UB_prop_graph_ori, self.usrEmbed, self.bndEmbed, "UB", self.UB_layer_coefs, test)
        else:
            UB_usrGEmb, UB_bndGEmb = self.propagate(self.UB_prop_graph_rnd, self.usrEmbed, self.bndEmbed, "UB", self.UB_layer_coefs, test)

        #  =============================  UI graph propagation  =============================
        if test:
            UI_usrGEmb, UI_itmGEmb = self.propagate(self.UI_prop_graph_ori, self.usrEmbed, self.itmEmbed, "UI", self.UI_layer_coefs, test)
            UI_bndGEmb = self.aggregate(self.BI_aggr_graph_ori, UI_itmGEmb, "BI", test)
        else:
            UI_usrGEmb, UI_itmGEmb = self.propagate(self.UI_prop_graph_rnd, self.usrEmbed, self.itmEmbed, "UI", self.UI_layer_coefs, test)
            UI_bndGEmb = self.aggregate(self.BI_aggr_graph_rnd, UI_itmGEmb, "BI", test)

        #  =============================  BI graph propagation  =============================
        if test:
            BI_bndGEmb, BI_itmGEmb = self.propagate(self.BI_prop_graph_ori, self.bndEmbed, self.itmEmbed, "BI", self.BI_layer_coefs, test)
            BI_usrGEmb = self.aggregate(self.UI_aggr_graph_ori, BI_itmGEmb, "UI", test)
        else:
            BI_bndGEmb, BI_itmGEmb = self.propagate(self.BI_prop_graph_rnd, self.bndEmbed, self.itmEmbed, "BI", self.BI_layer_coefs, test)
            BI_usrGEmb = self.aggregate(self.UI_aggr_graph_rnd, BI_itmGEmb, "UI", test)

        usrGEmb = [UB_usrGEmb, UI_usrGEmb, BI_usrGEmb]
        bndGEmb = [UB_bndGEmb, UI_bndGEmb, BI_bndGEmb]

        usr_rep, bnd_rep = self.fuse_usr_bnd_feat(usrGEmb, bndGEmb)

        return usr_rep, bnd_rep, usrGEmb, bndGEmb
    
    def cal_c_loss_0th(self, agg_emb, emb1, emb2, emb3):
        loss1 = self.cal_c0_loss_pair(emb1, agg_emb)
        loss2 = self.cal_c0_loss_pair(emb2, agg_emb)
        loss3 = self.cal_c0_loss_pair(emb3, agg_emb)
        loss= (loss1+loss2+loss3)/3
        return loss

    def cal_c0_loss_pair(self, pos, aug):
        pos = F.normalize(pos, p=2, dim=-1)
        aug = F.normalize(aug, p=2, dim=-1)
        
        pos_score = torch.sum(pos * aug, dim=1) # [batch_size]
        ttl_score = torch.matmul(pos, aug.permute(1, 0)) # [batch_size, batch_size]

        pos_score = torch.exp(pos_score / self.c_temp) # [batch_size]
        ttl_score = torch.sum(torch.exp(ttl_score / self.c_temp), dim=1) # [batch_size]

        c_loss = - torch.mean(torch.log(pos_score / ttl_score))
        return c_loss
        
    
    def cal_c_loss_1st(self, foc, aug):
        foc = F.normalize(foc, p=2, dim=-1)
        aug = F.normalize(aug, p=2, dim=-1)
        pos = aug[:, 0, :]
        
        pos_score = torch.sum(foc * pos, dim=1) # [batch_size]
        ttl_score = torch.matmul(aug, foc.unsqueeze(2)).squeeze(2) # [batch_size, batch_size]

        pos_score = torch.exp(pos_score / self.c_temp) # [batch_size]
        ttl_score = torch.sum(torch.exp(ttl_score / self.c_temp), dim=1) # [batch_size]

        c_loss = - torch.mean(torch.log(pos_score / ttl_score))
        return c_loss
    

    def cal_c_loss_2nd(self, foc, aug, num_nei):
        foc = F.normalize(foc, p=2, dim=-1)
        aug = F.normalize(aug, p=2, dim=-1)
        pos = aug[:, 0:num_nei, :]
        neg = aug[:, num_nei:, :]
        
        pos_score = torch.matmul(pos, foc.unsqueeze(2)).squeeze(2)#(batch_size, num_pos)
        neg_score = torch.matmul(neg, foc.unsqueeze(2)).squeeze(2)#(batch_size, num_neg)
        
        pos_score = torch.exp(pos_score / self.c_temp) # [batch_size, num_nei]
        neg_score = torch.sum(torch.exp(neg_score / self.c_temp), dim=1, keepdim=True)# [batch_size, 1]
        ttl_score = pos_score + neg_score
        c_loss = -torch.mean(torch.log(pos_score / ttl_score))
        
        return c_loss
        
    def bnd_prediction_singleview(self, u_idx, view):
        _, _, usrGEmb, bndGEmb = self.get_multi_modal_representations(test=True)
        UB_usrGEmb, UI_usrGEmb, BI_usrGEmb = usrGEmb
        UB_bndGEmb, UI_bndGEmb, BI_bndGEmb = bndGEmb

        if view == 'ub':
            u_embs = UB_usrGEmb[torch.LongTensor(u_idx).cuda()]
            preds = torch.matmul(u_embs, UB_bndGEmb.T)
        elif view == 'ui':
            u_embs = UI_usrGEmb[torch.LongTensor(u_idx).cuda()]
            preds = torch.matmul(u_embs, UI_bndGEmb.T)
        elif view == 'bi':
            u_embs = BI_usrGEmb[torch.LongTensor(u_idx).cuda()]
            preds = torch.matmul(u_embs, BI_bndGEmb.T)

        return preds

    
    def bnd_prediction(self, u_idx):
        usr_rep, bnd_rep,_,_ = self.get_multi_modal_representations(test=True)

        u_embs = usr_rep[torch.LongTensor(u_idx).cuda()]
        preds = torch.matmul(u_embs, bnd_rep.T)
        return preds
        
    def train_bnd_loss(self, u_idx, b_idx, u_neis, b_neis):
        # the edge drop can be performed by every batch or epoch, should be controlled in the train loop
        if self.aug_type== 'ED':
            self.UB_prop_graph_rnd = self.get_propagation_graph(self.ub_graph, self.UB_ratio)

            self.UI_prop_graph_rnd = self.get_propagation_graph(self.ui_graph, self.UI_ratio)
            self.UI_aggr_graph_rnd = self.get_aggregation_graph(self.ui_graph, self.UI_ratio)

            self.BI_prop_graph_rnd = self.get_propagation_graph(self.bi_graph, self.BI_ratio)
            self.BI_aggr_graph_rnd = self.get_aggregation_graph(self.bi_graph, self.BI_ratio)
        
        usr_rep, bnd_rep, usrGEmb, bndGEmb = self.get_multi_modal_representations()
        u_embs = usr_rep[u_idx]
        b_embs = bnd_rep[b_idx]
        
        u_bpr_embs = u_embs[:, 0, :]
        b_bpr_embs = b_embs[:, 0:2, :]
        
        preds = torch.matmul(b_bpr_embs, u_bpr_embs.unsqueeze(2)).squeeze(2)
        bpr_loss = cal_bpr_loss(preds)

        upos_embs = u_bpr_embs
        bpos_embs = b_bpr_embs[:, 0, :]
        
        UB_usrGEmb, UI_usrGEmb, BI_usrGEmb = usrGEmb
        UB_bndGEmb, UI_bndGEmb, BI_bndGEmb = bndGEmb
        
        UB_u_embs, UI_u_embs, BI_u_embs = UB_usrGEmb[u_idx[:, 0]], UI_usrGEmb[u_idx[:, 0]], BI_usrGEmb[u_idx[:, 0]]
        UB_b_embs, UI_b_embs, BI_b_embs = UB_bndGEmb[b_idx[:, 0]], UI_bndGEmb[b_idx[:, 0]], BI_bndGEmb[b_idx[:, 0]]
        
        c0_loss1 = self.cal_c_loss_0th(upos_embs, UB_u_embs, UI_u_embs, BI_u_embs)
        c0_loss2 = self.cal_c_loss_0th(bpos_embs, UB_b_embs, UI_b_embs, BI_b_embs)
        c0_loss = 0.5*(c0_loss1 + c0_loss2)
        
        c1_loss1 = self.cal_c_loss_1st(upos_embs, b_embs)
        c1_loss2 = self.cal_c_loss_1st(bpos_embs, u_embs)
        c1_loss = 0.5*(c1_loss1 + c1_loss2)
        
        unei_embs = usr_rep[u_neis]
        bnei_embs = bnd_rep[b_neis]
        c2_loss1 = self.cal_c_loss_2nd(upos_embs, unei_embs, self.num_neigh)
        c2_loss2 = self.cal_c_loss_2nd(bpos_embs, bnei_embs, self.num_neigh)
        c2_loss = 0.5*(c2_loss1 + c2_loss2)
        loss = bpr_loss + self.c0_reg * c0_loss + self.c1_reg * c1_loss + self.c2_reg * c2_loss
        return loss
    

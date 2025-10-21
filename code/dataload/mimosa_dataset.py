import torch
from torch.utils.data import Dataset, DataLoader
from dataload.datautils import *

class ItemTrainDataset(Dataset):
    def __init__(self, ui_pairs, ui_graph, num_itm, num_neg=1):
        self.ui_pairs = ui_pairs
        self.ui_graph = ui_graph
        self.num_itm = num_itm
        self.num_neg = num_neg
        
    def __getitem__(self, index):
        usr, pos_itm = self.ui_pairs[index]
        all_itm = [pos_itm]
        for _ in range(self.num_neg):
            i = np.random.randint(self.num_itm)
            while self.ui_graph[usr, i] == 1 or i in all_itm:
                i = np.random.randint(self.num_itm)
            all_itm.append(i)                                                                                                         
        return torch.LongTensor([usr]), torch.LongTensor(all_itm)

    def __len__(self):
        return len(self.ui_pairs)
        
class BundleTrainDataset(Dataset):
    def __init__(self, ub_pairs, ub_graph, u_neis, b_neis, num_usr, num_bnd, num_neg=1):
        self.ub_pairs = ub_pairs
        self.ub_graph = ub_graph
        self.u_neis = u_neis
        self.b_neis = b_neis
        self.num_usr = num_usr
        self.num_bnd = num_bnd
        self.num_neg = num_neg

    def __getitem__(self, index):
        pos_usr, pos_bnd = self.ub_pairs[index]
        all_bnd = [pos_bnd]
        for _ in range(self.num_neg):
            i = np.random.randint(self.num_bnd)
            while self.ub_graph[pos_usr, i] == 1 or i in all_bnd:
                i = np.random.randint(self.num_bnd)
            all_bnd.append(i) 
        all_usr = [pos_usr]
        for _ in range(self.num_neg):
            i = np.random.randint(self.num_usr)
            while self.ub_graph[i, pos_bnd] == 1 or i in all_usr:
                i = np.random.randint(self.num_usr)
            all_usr.append(i)
            
        all_u_neis = [v for v in self.u_neis[pos_usr]]
        for _ in range(self.num_neg):
            i = np.random.randint(self.num_usr)
            while i in all_u_neis:
                i = np.random.randint(self.num_usr)
            all_u_neis.append(i)
        all_b_neis = [v for v in self.b_neis[pos_bnd]]
        for _ in range(self.num_neg):
            i = np.random.randint(self.num_bnd)
            while i in all_b_neis:
                i = np.random.randint(self.num_bnd)
            all_b_neis.append(i)
        return torch.LongTensor(all_usr), torch.LongTensor(all_bnd), torch.LongTensor(all_u_neis), torch.LongTensor(all_b_neis)

    def __len__(self):
        return len(self.ub_pairs)

class BundleEvalDataset(Dataset):
    def __init__(self, ub_pairs, ub_graph_eval, ub_graph_train, num_usr, num_bnd):
        self.ub_pairs = ub_pairs
        self.ub_graph = ub_graph_eval
        self.train_mask_ub = ub_graph_train

        self.usr = torch.arange(num_usr, dtype=torch.long).unsqueeze(dim=1)
        self.bnd = torch.arange(num_bnd, dtype=torch.long)

    def __getitem__(self, index):
        ub_grd = torch.from_numpy(self.ub_graph[index].toarray()).squeeze()#ground truth
        ub_mask = torch.from_numpy(self.train_mask_ub[index].toarray()).squeeze()#training data that should be masked in the test stage.

        return index, ub_grd, ub_mask

    def __len__(self):
        return self.ub_graph.shape[0]
        
class BundleDataset(object):
    def __init__(self, data, num_neg, num_neigh):
        data_path = '../data/{}/'.format(data)
        self.num_neg = num_neg
        self.num_neigh = num_neigh
        self.num_usr, self.num_bnd, self.num_itm = self.get_data_size(data_path, data)

        self.bi_pairs, self.bi_graph = get_bi(data_path, self.num_bnd, self.num_itm)
        self.ui_pairs, self.ui_graph = get_ui(data_path, self.num_usr, self.num_itm)

        self.ub_pairs_train, self.ub_graph_train = get_ub(data_path, "train", self.num_usr, self.num_bnd)
        self.ub_pairs_val, self.ub_graph_val = get_ub(data_path, "tune", self.num_usr, self.num_bnd)
        self.ub_pairs_test, self.ub_graph_test = get_ub(data_path, "test", self.num_usr, self.num_bnd)
        self.graphs = [self.ub_graph_train, self.ui_graph, self.bi_graph]
        
        self.b_neis = self.get_ii_constraint_mat(self.ub_graph_train, self.num_neigh)
        self.u_neis = self.get_ii_constraint_mat(self.ub_graph_train.T, self.num_neigh)
        
    def get_data_size(self, path, data):
        name = data
        if "_" in name:
            name = name.split("_")[0]
        with open(os.path.join(path, '{}_data_size.txt'.format(name)), 'r') as f:
            return [int(s) for s in f.readline().split('\t')][:3]
        
    def get_ii_constraint_mat(self,train_mat, num_neighbors, ii_diagonal_zero=True):
        print('Computing \\Omega for the item-item graph... ')
        A = train_mat.T.dot(train_mat)  # B * B
        A_tensor = torch.from_numpy(A.toarray())
        _, row_idxs = torch.topk(A_tensor, num_neighbors)
        print ('row:{}'.format(row_idxs.shape))
        return torch.LongTensor(row_idxs)
        
    def get_train_bnd_loader(self, batch_size):
        bnd_train_data = BundleTrainDataset(self.ub_pairs_train, self.ub_graph_train, self.u_neis, self.b_neis, self.num_usr, self.num_bnd, self.num_neg)
        train_bnd_loader = DataLoader(bnd_train_data, batch_size=batch_size, shuffle=True, num_workers=10, drop_last=True)
        return train_bnd_loader
        
    def get_val_bnd_loader(self, batch_size):
        bnd_val_data = BundleEvalDataset(self.ub_pairs_val, self.ub_graph_val, self.ub_graph_train, self.num_usr, self.num_bnd)
        val_bnd_loader = DataLoader(bnd_val_data, batch_size=batch_size, shuffle=False, num_workers=20)
        return val_bnd_loader
        
    def get_test_bnd_loader(self, batch_size):
        bnd_test_data = BundleEvalDataset(self.ub_pairs_test, self.ub_graph_test, self.ub_graph_train, self.num_usr, self.num_bnd)
        test_bnd_loader = DataLoader(bnd_test_data, batch_size=batch_size, shuffle=False, num_workers=20)
        return test_bnd_loader

import matplotlib.pyplot as plt
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import scipy.sparse
import torch
from torch import Tensor
from core.search_space import SearchSpace
from optimizer_hubs import GA, PSO, SSA, GWO, SO, WOA
from torch_geometric.data import Data
from core.build_network import GraphNet
from core.utiles import *
from core.components import optim_build
from torch.optim.lr_scheduler import StepLR
from core.train_val_test import cal_trainset, cal_testset
from sklearn.utils import resample
import torch
import numpy as np
from torch_geometric.data import DataLoader
from torch_geometric.utils.num_nodes import maybe_num_nodes


def to_scipy_sparse_matrix(
    edge_index: Tensor,
    edge_attr: Optional[Tensor] = None,
    num_nodes: Optional[int] = None,
) -> scipy.sparse.coo_matrix:
    r"""Converts a graph given by edge indices and edge attributes to a scipy
    sparse matrix.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional
            edge features. (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)

    Examples:

        >>> edge_index = torch.tensor([
        ...     [0, 1, 1, 2, 2, 3],
        ...     [1, 0, 2, 1, 3, 2],
        ... ])
        >>> to_scipy_sparse_matrix(edge_index)
        <4x4 sparse matrix of type '<class 'numpy.float32'>'
            with 6 stored elements in COOrdinate format>
    """
    row, col = edge_index.cpu()

    if edge_attr is None:
        edge_attr = torch.ones(row.size(0))
    else:
        edge_attr = edge_attr.view(-1).cpu()
        assert edge_attr.size(0) == row.size(0)

    N = maybe_num_nodes(edge_index, num_nodes)
    out = scipy.sparse.coo_matrix(
        (edge_attr.numpy(), (row.numpy(), col.numpy())), (N, N))
    return out

class IOA(object):
    def __init__(self, args):
        self.args = args
        self.se_sp = SearchSpace(self.args.num_gnn_layers)
        self.x_train, self.x_test, self.y_train, self.y_test, self.edge_index =\
            self.data_preprocess(self.args.site)

    def start_search(self):
        if self.args.ioa == 'ga':
            fmdl = GA.ga(self.x_train, self.y_train, self.edge_index, self.args, self.se_sp)
        elif self.args.ioa == 'pso':
            fmdl = PSO.pso(self.x_train, self.y_train, self.edge_index, self.args, self.se_sp)
        elif self.args.ioa == 'ssa':
            fmdl = SSA.ssa(self.x_train, self.y_train, self.edge_index, self.args, self.se_sp)
        elif self.args.ioa == 'gwo':
            fmdl = GWO.gwo(self.x_train, self.y_train, self.edge_index, self.args, self.se_sp)
        elif self.args.ioa == 'so':
            fmdl = SO.so(self.x_train, self.y_train, self.edge_index, self.args, self.se_sp)
        elif self.args.ioa == 'woa':
            fmdl = WOA.woa(self.x_train, self.y_train, self.edge_index, self.args, self.se_sp)
        else:
            raise NotImplementedError
        return fmdl['best_structure'], fmdl['best_fitness']

    def train_test_explain_one_structure(self, structure):
        train_data_list, test_data_list = [], []
        for i in range(self.x_train.shape[0]):
            x = torch.tensor(self.x_train[i], dtype=torch.float)
            data = Data(x=x, edge_index=self.edge_index, y=torch.tensor(self.y_train[i], dtype=torch.long))
            train_data_list.append(data)
        for i in range(self.x_test.shape[0]):
            x = torch.tensor(self.x_test[i], dtype=torch.float)  # 节点特征
            data = Data(x=x, edge_index=self.edge_index, y=torch.tensor(self.y_test[i], dtype=torch.long))
            test_data_list.append(data)
        train_loader = DataLoader(train_data_list, batch_size=64, shuffle=False)
        test_loader = DataLoader(test_data_list, batch_size=64, shuffle=False)

        model = GraphNet(self.args.num_gnn_layers, structure[:-3], input_feature=140, explainable=True)
        device = torch.device(self.args.cuda)
        model = model.to(device)
        optimizer = optim_build(model, structure[-3:])
        criterion = torch.nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=1)
        iters = len(train_loader)
        best_acc, best_y = 0, None
        for e in range(1, 10):
            fine_train(model, train_loader, optimizer, criterion, device, e, iters, scheduler)
            train_acc = cal_trainset(model, train_loader, device)
            best_y, best_acc = cal_testset(model, test_loader, device, best_acc, best_y)
        # print(best_metrics)

        correct_predictions_label_0 = []
        correct_predictions_label_1 = []
        target = best_y.argmax(dim=1)

        for data in test_loader:
            data = data.to(device)
        for i in range(len(data)):
            if target[i] == data.y[i]:
                if target[i] == 0:
                    correct_predictions_label_0.append(data[i])
                elif target[i] == 1:
                    correct_predictions_label_1.append(data[i])
        label_0_explain_load = DataLoader(correct_predictions_label_0, batch_size=1, shuffle=False)
        label_1_explain_load = DataLoader(correct_predictions_label_1, batch_size=1, shuffle=False)
        # 找到分类正确的，最后求mask的均值
        import pandas as pd
        from explain.gnnexplainer import GNNExplainer
        exp = GNNExplainer(model)
        edge_mask_list = []
        node_mask_list = []
        for data in label_0_explain_load:
            node_mask, edge_mask = exp.explain_graph(data.x, data.edge_index)
            edge_mask_list.append(edge_mask)
            node_mask_list.append(node_mask)
            # exp.visualize_subgraph(-1, data.edge_index, edge_mask, threshold=0.5)
            # plt.show()
        label_0_mean_edge_mask = np.mean(np.stack(edge_mask_list, axis=0), axis=0)
        label_0_mean_node_mask = np.mean(np.stack(node_mask_list, axis=0), axis=0)
        df = pd.DataFrame(label_0_mean_node_mask, index=range(1, 141), columns=['Data'])
        # 对数据进行降序排列
        df_sorted = df.sort_values(by='Data', ascending=False)
        # 获取排列后的前5个数据及其索引
        top_5 = df_sorted.head(5)
        top_5_indices = top_5.index.tolist()
        top_5_values = top_5['Data'].tolist()
        print(top_5_indices, top_5_values)
        # 对数据进行排序，从大到小
        sorted_data = np.sort(label_0_mean_edge_mask)[::-1]
        # 计算前10%的数量
        top_10_percent_count = int(len(label_0_mean_edge_mask) * 0.03)
        # 获取前10%值最大的元素
        top_10_percent_values = sorted_data[:top_10_percent_count]
        # 获取这些元素在原始数据中的索引
        label_0_top_10_percent_indices = [np.where(label_0_mean_edge_mask == value)[0][0] for value in top_10_percent_values]
        # label_0_edge_index = data.edge_index[:, label_0_top_10_percent_indices]
        # from torch_geometric.utils import to_dense_adj
        import torch_geometric
        out = torch_geometric.utils.to_dense_adj(data.edge_index[:, label_0_top_10_percent_indices])
        file_name1 = 'ga_label_0_edge_index_1.csv'
        file_name2 = 'ga_label_0_node_index_1.csv'
        np.savetxt(file_name1, np.array(out.data[0,:,:]), delimiter=',')
        np.savetxt(file_name2, label_0_mean_node_mask, delimiter=',')

        edge_mask_list = []
        node_mask_list = []
        for data in label_1_explain_load:
            node_mask, edge_mask = exp.explain_graph(data.x, data.edge_index)
            edge_mask_list.append(edge_mask)
            node_mask_list.append(node_mask)
        label_1_mean_edge_mask = np.mean(np.stack(edge_mask_list, axis=0), axis=0)
        label_1_mean_node_mask = np.mean(np.stack(node_mask_list, axis=0), axis=0)
        df = pd.DataFrame(label_1_mean_node_mask, index=range(1, 141), columns=['Data'])
        # 对数据进行降序排列
        df_sorted = df.sort_values(by='Data', ascending=False)
        # 获取排列后的前5个数据及其索引
        top_5 = df_sorted.head(5)
        top_5_indices = top_5.index.tolist()
        top_5_values = top_5['Data'].tolist()
        print(top_5_indices, top_5_values)
        # 对数据进行排序，从大到小
        sorted_data = np.sort(label_1_mean_edge_mask)[::-1]
        # 计算前10%的数量
        top_10_percent_count = int(len(label_1_mean_edge_mask) * 0.03)
        # 获取前10%值最大的元素
        top_10_percent_values = sorted_data[:top_10_percent_count]
        # 获取这些元素在原始数据中的索引
        label_1_top_10_percent_indices = [np.where(label_1_mean_edge_mask == value)[0][0] for value in top_10_percent_values]

        out = torch_geometric.utils.to_dense_adj(data.edge_index[:, label_1_top_10_percent_indices])
        # 如果你想要的是稀疏的邻接矩阵，可以使用以下方法
        # sparse_adjacency_matrix = adjacency_matrix.to_sparse(140)
        file_name1 = 'ga_label_1_edge_index.csv'
        file_name2 = 'ga_label_1_node_index.csv'
        np.savetxt(file_name1, np.array(out.data[0,:,:]), delimiter=',')
        np.savetxt(file_name2, label_1_mean_node_mask, delimiter=',')

    def data_preprocess(self, site='all'):
        HC, SCH = self.SCH_HC(site=site, Balance_flag=True)
        HC_weighted, SCH_weighted = map_to_triangle(HC), map_to_triangle(SCH)

        HC_label, SCH_label = \
            np.zeros(HC_weighted.shape[0], dtype=int), np.ones(SCH_weighted.shape[0], dtype=int)
        X = np.concatenate([HC_weighted, SCH_weighted], axis=0)
        y = np.concatenate([HC_label, SCH_label], axis=0)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

        Connectome_weighted = np.mean(X_train, axis=0)
        Connectome_weighted = mst_threshold(Connectome_weighted, MyCost=self.args.myCost)

        Connectome_binary = Connectome_weighted.copy()

        edge_index = create_edge_index_from_adjacency_matrix(Connectome_binary)

        return X_train, X_test, y_train, y_test, edge_index

    def SCH_HC(self, site='all', Balance_flag=True):
        KTT_HC = np.load('data/HC_SCH/KTT/HC.npy', allow_pickle=True)
        KTT_SCH = np.load('data/HC_SCH/KTT/SCH.npy', allow_pickle=True)
        KUT_HC = np.load('data/HC_SCH/KUT/HC.npy', allow_pickle=True)
        KUT_SCH = np.load('data/HC_SCH/KUT/SCH.npy', allow_pickle=True)
        SWA_HC = np.load('data/HC_SCH/SWA/HC.npy', allow_pickle=True)
        SWA_SCH = np.load('data/HC_SCH/SWA/SCH.npy', allow_pickle=True)
        UTO_HC = np.load('data/HC_SCH/UTO/HC.npy', allow_pickle=True)
        UTO_SCH = np.load('data/HC_SCH/UTO/SCH.npy', allow_pickle=True)
        # hc = np.load('data/HC_MDD/hc.npy', allow_pickle=True)
        # mdd = np.load('data/HC_MDD/mdd.npy', allow_pickle=True)
        if Balance_flag:
            KTT_HC = resample(KTT_HC, replace=False, n_samples=KTT_SCH.shape[0], random_state=67)
            KUT_HC = resample(KUT_HC, replace=False, n_samples=KUT_SCH.shape[0], random_state=67)
            SWA_HC = resample(SWA_HC, replace=False, n_samples=SWA_SCH.shape[0], random_state=67)
            UTO_HC = resample(UTO_HC, replace=False, n_samples=UTO_SCH.shape[0], random_state=67)
            # hc = resample(hc, replace=False, n_samples=mdd.shape[0], random_state=67)
        # SWA_HC = np.load('data/HC_MDD/hc.npy', allow_pickle=True)
        # SWA_MDD = np.load('data/HC_MDD/mdd.npy', allow_pickle=True)
        if site == 'all':
            HC = np.concatenate([KTT_HC, KUT_HC, SWA_HC, UTO_HC])
            SCH = np.concatenate([KTT_SCH, KUT_SCH, SWA_SCH, UTO_SCH])
        elif site == 'KTT':
            HC, SCH = KTT_HC, KTT_SCH
        elif site == 'KUT':
            HC, SCH = KUT_HC, KUT_SCH
        elif site == 'SWA':
            HC, SCH = SWA_HC, SWA_SCH
        elif site == 'UTO':
            HC, SCH = UTO_HC, UTO_SCH
        else:
            raise NotImplementedError
        return HC, SCH

    def cross_vali(self, codes):
        from sklearn.model_selection import StratifiedKFold
        xtrain, ytrain, edge_index, args = self.x_train, self.y_train, self.edge_index, self.args
        kf = StratifiedKFold(n_splits=5, shuffle=False)
        scores = []
        for fold, (train_idx, val_idx) in enumerate(kf.split(xtrain, ytrain)):
            train_data_list, val_data_list = [], []
            for i in range(xtrain[train_idx].shape[0]):
                x = torch.tensor(xtrain[train_idx][i], dtype=torch.float)
                data = Data(x=x, edge_index=edge_index, y=torch.tensor(ytrain[train_idx][i], dtype=torch.long))
                train_data_list.append(data)
            for i in range(xtrain[val_idx].shape[0]):
                x = torch.tensor(xtrain[val_idx][i], dtype=torch.float)  # 节点特征
                data = Data(x=x, edge_index=edge_index, y=torch.tensor(ytrain[val_idx][i], dtype=torch.long))
                val_data_list.append(data)
            train_loader = DataLoader(train_data_list, batch_size=64, shuffle=False)
            val_loader = DataLoader(val_data_list, batch_size=64, shuffle=False)
            model = GraphNet(args.num_gnn_layers, codes[:-3], input_feature=140, explainable=False)
            device = torch.device(args.cuda)
            model = model.to(device)
            optimizer = optim_build(model, codes[-3:])
            criterion = torch.nn.CrossEntropyLoss()
            best_val_accuracy = 0
            for e in range(1, args.epochs):
                train(model, train_loader, optimizer, criterion, device)
                val_loss, val_acc = test(model, val_loader, criterion, device)
                if val_acc > best_val_accuracy:
                    best_val_accuracy = val_acc
                # print(e)
                # print(val_acc)
            scores.append(best_val_accuracy)
        print(scores)
        # print(fitness)







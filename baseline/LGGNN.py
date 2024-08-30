"""
Author: Shuning Han

This module defines a Functional connectom（FC）based（Graph Neural Network (GNN) model for fMRI analysis using PyTorch Geometric.

"""
from torch_geometric.nn import Linear,ChebConv, global_mean_pool, global_max_pool, GCNConv, GATConv, GraphSAGE, BatchNorm

import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import ChebConv


class Global_GNN(nn.Module):
    def __init__(self):
        super(Global_GNN, self).__init__()
        self.num_layers = 4
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.convs.append(ChebConv(140, 20,K=3,normalization='sym'))
        self.bns.append(nn.BatchNorm1d(20))
        self.convs.append(ChebConv(20, 20,K=3,normalization='sym'))
        self.bns.append(nn.BatchNorm1d(20))
        self.convs.append(ChebConv(20, 20,K=3,normalization='sym'))
        self.bns.append(nn.BatchNorm1d(20))
        self.convs.append(ChebConv(20, 20,K=3,normalization='sym'))
        self.bns.append(nn.BatchNorm1d(20))
        self.out_fc = nn.Linear(20, 2)
        self.weights = torch.nn.Parameter(torch.randn((len(self.convs))))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.out_fc.reset_parameters()
        torch.nn.init.normal_(self.weights)

    def forward(self, x, edge_index, batch):
        edges = edge_index
        layer_out = []
        x = self.convs[0](x, edges)
        x = self.bns[0](x)
        x = F.relu(x, inplace=True)
        layer_out.append(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.convs[1](x, edges)
        x = self.bns[1](x)
        x = F.relu(x, inplace=True)
        x = x +  0.7 * layer_out[0]
        layer_out.append(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.convs[2](x, edges)
        x = self.bns[2](x)
        x = F.relu(x, inplace=True)
        x = x +  0.7 * layer_out[1]
        layer_out.append(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.convs[3](x, edges)
        x = self.bns[3](x)
        x = F.relu(x, inplace=True)
        x = x +  0.7 * layer_out[2]
        layer_out.append(x)
        weight = F.softmax(self.weights, dim=0)
        for i in range(len(layer_out)):
            layer_out[i] = layer_out[i] * weight[i]
        # Global Mean Pooling
        emb = sum(layer_out)
        x = global_mean_pool(emb, batch)
        # Dropout
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.out_fc(x)
        return x



from torch_geometric.data import Data
from core.utiles import *
from sklearn.utils import resample
import torch
import numpy as np
from torch_geometric.data import DataLoader

def SCH_HC(site='all', Balance_flag=True):
    KTT_HC = np.load('/Users/wangsr/PycharmProjects/graph_fc/data/HC_SCH/KTT/HC.npy', allow_pickle=True)
    KTT_SCH = np.load('/Users/wangsr/PycharmProjects/graph_fc/data/HC_SCH/KTT/SCH.npy', allow_pickle=True)
    KUT_HC = np.load('/Users/wangsr/PycharmProjects/graph_fc/data/HC_SCH/KUT/HC.npy', allow_pickle=True)
    KUT_SCH = np.load('/Users/wangsr/PycharmProjects/graph_fc/data/HC_SCH/KUT/SCH.npy', allow_pickle=True)
    SWA_HC = np.load('/Users/wangsr/PycharmProjects/graph_fc/data/HC_SCH/SWA/HC.npy', allow_pickle=True)
    SWA_SCH = np.load('/Users/wangsr/PycharmProjects/graph_fc/data/HC_SCH/SWA/SCH.npy', allow_pickle=True)
    UTO_HC = np.load('/Users/wangsr/PycharmProjects/graph_fc/data/HC_SCH/UTO/HC.npy', allow_pickle=True)
    UTO_SCH = np.load('/Users/wangsr/PycharmProjects/graph_fc/data/HC_SCH/UTO/SCH.npy', allow_pickle=True)
    if Balance_flag:
        KTT_HC = resample(KTT_HC, replace=False, n_samples=KTT_SCH.shape[0], random_state=67)
        KUT_HC = resample(KUT_HC, replace=False, n_samples=KUT_SCH.shape[0], random_state=67)
        SWA_HC = resample(SWA_HC, replace=False, n_samples=SWA_SCH.shape[0], random_state=67)
        UTO_HC = resample(UTO_HC, replace=False, n_samples=UTO_SCH.shape[0], random_state=67)
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

seed_everything(6767) #3
HC, SCH = SCH_HC(site='all', Balance_flag=True)
HC_weighted, SCH_weighted = map_to_triangle(HC), map_to_triangle(SCH)
HC_label, SCH_label = np.zeros(HC_weighted.shape[0], dtype=int), np.ones(SCH_weighted.shape[0], dtype=int)
X = np.concatenate([HC_weighted, SCH_weighted], axis=0)
y = np.concatenate([HC_label, SCH_label], axis=0)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
Connectome_weighted = np.mean(x_train, axis=0)
# Connectome_weighted = mst_threshold(Connectome_weighted, MyCost=0.2)
Connectome_binary = Connectome_weighted.copy()
edge_index = create_edge_index_from_adjacency_matrix(Connectome_binary)

train_data_list, test_data_list = [], []
for i in range(x_train.shape[0]):
    x = torch.tensor(x_train[i], dtype=torch.float)
    data = Data(x=x, edge_index=edge_index, y=torch.tensor(y_train[i], dtype=torch.long))
    train_data_list.append(data)
for i in range(x_test.shape[0]):
    x = torch.tensor(x_test[i], dtype=torch.float)  # 节点特征
    data = Data(x=x, edge_index=edge_index, y=torch.tensor(y_test[i], dtype=torch.long))
    test_data_list.append(data)
train_loader = DataLoader(train_data_list, batch_size=64, shuffle=False)
test_loader = DataLoader(test_data_list, batch_size=64, shuffle=False)

device = torch.device('cpu')
model = Global_GNN().to(device)
import torch.optim as op
optimizer = op.Adam(params=model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=1)
best_test_accuracy = 0
iters = len(train_loader)
for e in range(1, 30):
    fine_train(model, train_loader, optimizer, criterion, device, e, iters, scheduler)
    print('epoch:{0}'.format(e))
    train_loss, train_acc = test(model, train_loader, criterion, device)
    print(f'Train Loss:{train_loss}, Correct Predictions: {train_acc}')
    test_loss, test_acc = test(model, test_loader, criterion, device)
    print(f'Test Loss:{test_loss}, Correct Predictions: {test_acc}')


from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
for data in test_loader:
    data = data.to(device)
    y_pred = model(data.x, data.edge_index, data.batch)
    y_pred = y_pred.detach().cpu().numpy()
    y = data.y.detach().cpu().numpy()
    f1 = f1_score(y, y_pred.argmax(axis=1))
    precision = precision_score(y, y_pred.argmax(axis=1))
    auc = roc_auc_score(y, y_pred.argmax(axis=1))
    accuracy = accuracy_score(y, y_pred.argmax(axis=1))
    recall_score = recall_score(y, y_pred.argmax(axis=1))

print('accuracy', accuracy)
print('precision', precision)
print('recall_score', recall_score)
print('f1', f1)
print('auc', auc)

torch.save(model, 'LGGNN.pt')
model = torch.load('LGGNN.pt')

label_0 = []
label_1 = []
target = y_pred.argmax(axis=1)
correct_predictions_label_0 = []
correct_predictions_label_1 = []
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

from explain.gnnexplainer import GNNExplainer
import pandas as pd
exp = GNNExplainer(model)
node_mask_list = []
for data in label_0_explain_load:
    node_mask, edge_mask = exp.explain_graph(data.x, data.edge_index)
    node_mask_list.append(node_mask.cpu())
label_0_mean_node_mask = np.mean(np.stack(node_mask_list, axis=0), axis=0)
df = pd.DataFrame(label_0_mean_node_mask, index=range(1, 141), columns=['Data'])
# 对数据进行降序排列
df_sorted = df.sort_values(by='Data', ascending=False)
# 获取排列后的前5个数据及其索引
top_5 = df_sorted.head(5)
top_5_indices = top_5.index.tolist()
top_5_values = top_5['Data'].tolist()
print(top_5_indices, top_5_values)


node_mask_list = []
for data in label_1_explain_load:
    node_mask, edge_mask = exp.explain_graph(data.x, data.edge_index)
    node_mask_list.append(node_mask.cpu())
# node_mask, edge_mask = exp.explain_graph(label_0[0].x, label_0[0].edge_index)
# 创建DataFrame，索引设置为1到10
label_1_mean_node_mask = np.mean(np.stack(node_mask_list, axis=0), axis=0)
df = pd.DataFrame(label_1_mean_node_mask, index=range(1, 141), columns=['Data'])
# 对数据进行降序排列
df_sorted = df.sort_values(by='Data', ascending=False)
# 获取排列后的前5个数据及其索引
top_5 = df_sorted.head(5)
top_5_indices = top_5.index.tolist()
top_5_values = top_5['Data'].tolist()
print(top_5_indices, top_5_values)



# model = Global_GNN().to(device)
# import torch.optim as op
# optimizer = op.Adam(params=model.parameters(), lr=0.01)
# criterion = torch.nn.CrossEntropyLoss()
# scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=1)
# best_test_accuracy = 0
# iters = len(train_loader)
# best_acc, best_metrics, best_y = 0, {}, None
#
# for e in range(1, 10):
#     fine_train(model, train_loader, optimizer, criterion, device, e, iters, scheduler)
#     train_acc = cal_trainset(model, train_loader, device)
#     best_y, best_metrics, best_acc = cal_testset(model, test_loader, device, best_acc, best_metrics, best_y)
# print(best_metrics)

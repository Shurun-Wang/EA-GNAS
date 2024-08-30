import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from core.train_val_test import train, test
from core.build_network import GraphNet
from core.components import *


# Error rate & Feature size
def Fun(xtrain, ytrain, edge_index, codes, args):
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
    fitness = 1-np.mean(scores)
    # print(fitness)
    return fitness
    # if codes not in global_structure:
    #     model = GraphNet(args.num_gnn_layers, codes[:-3], input_feature=140, explainable=False)
    #     device = torch.device(args.cuda)
    #     model = model.to(device)
    #     optimizer = optim_build(model, codes[-3:])
    #     criterion = torch.nn.CrossEntropyLoss()
    #     best_val_accuracy = 0
    #     for e in range(1, args.epochs):
    #         train(model, train_loader, optimizer, criterion, device)
    #         val_loss, val_acc = test(model, val_loader, criterion, device)
    #         if val_acc > best_val_accuracy:
    #             best_val_accuracy = val_acc
    #             # torch.save(model.state_dict(), 'checkpoints/ga/model'+str(len(global_structure))1.pth')')
    #             # print(val_acc)
    #         print(e)
    #         print(val_acc)
    #     fitness = 1 - best_val_accuracy
    #     print(fitness)
    #     print(codes)
    #     global_structure.append(codes)
    #     global_fitness.append(fitness)
    # else:
    #     index = global_structure.index(codes)
    #     fitness = global_fitness[index]
    # # print(1-np.nanmean(scores))
    # return fitness, global_fitness, global_structure


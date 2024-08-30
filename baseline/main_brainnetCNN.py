import torch.nn
import torch.nn
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset
from torch_geometric.data import DataLoader
from core.utiles import *
from sklearn.utils import resample
class E2EBlock(torch.nn.Module):
    '''E2Eblock.'''
    def __init__(self, in_planes, planes, example, bias=False):
        super(E2EBlock, self).__init__()
        # self.d = example.size(3)
        self.d = 140
        self.cnn1 = torch.nn.Conv2d(in_planes, planes, (1, self.d), bias=bias)
        self.cnn2 = torch.nn.Conv2d(in_planes, planes, (self.d, 1), bias=bias)

    def forward(self, x):
        a = self.cnn1(x)
        b = self.cnn2(x)
        return torch.cat([a] * self.d, 3) + torch.cat([b] * self.d, 2)

import torch.nn.functional as F
class BrainNetCNN(torch.nn.Module):
    def __init__(self, num_classes=2):
        super(BrainNetCNN, self).__init__()
        # self.d = example.size(3)
        self.d = 140
        self.e2econv1 = E2EBlock(1, 32, 140, bias=True)
        self.e2econv2 = E2EBlock(32, 64, 140, bias=True)
        self.E2N = torch.nn.Conv2d(64, 1, (1, self.d))
        self.N2G = torch.nn.Conv2d(1, 256, (self.d, 1))
        self.dense1 = torch.nn.Linear(256, 128)
        self.dense2 = torch.nn.Linear(128, 30)
        self.dense3 = torch.nn.Linear(30, 2)

    def forward(self, x):
        out = F.leaky_relu(self.e2econv1(x), negative_slope=0.33)
        out = F.leaky_relu(self.e2econv2(out), negative_slope=0.33)
        out = F.leaky_relu(self.E2N(out), negative_slope=0.33)
        out = F.dropout(F.leaky_relu(self.N2G(out), negative_slope=0.33), p=0.5)
        out = out.view(out.size(0), -1)
        out = F.dropout(F.leaky_relu(self.dense1(out), negative_slope=0.33), p=0.5)
        out = F.dropout(F.leaky_relu(self.dense2(out), negative_slope=0.33), p=0.5)
        out = F.leaky_relu(self.dense3(out), negative_slope=0.33)

        return out

def SCH_HC(site='all', Balance_flag=True):
    KTT_HC = np.load('../data/HC_SCH/KTT/HC.npy', allow_pickle=True)
    KTT_SCH = np.load('../data/HC_SCH/KTT/SCH.npy', allow_pickle=True)
    KUT_HC = np.load('../data/HC_SCH/KUT/HC.npy', allow_pickle=True)
    KUT_SCH = np.load('../data/HC_SCH/KUT/SCH.npy', allow_pickle=True)
    SWA_HC = np.load('../data/HC_SCH/SWA/HC.npy', allow_pickle=True)
    SWA_SCH = np.load('../data/HC_SCH/SWA/SCH.npy', allow_pickle=True)
    UTO_HC = np.load('../data/HC_SCH/UTO/HC.npy', allow_pickle=True)
    UTO_SCH = np.load('../data/HC_SCH/UTO/SCH.npy', allow_pickle=True)
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


seed_everything(6767)

HC, PA = SCH_HC(site='all', Balance_flag=True)
HC_weighted, SCH_weighted = map_to_triangle(HC), map_to_triangle(PA)

HC_label, SCH_label = \
    np.zeros(HC_weighted.shape[0], dtype=int), np.ones(SCH_weighted.shape[0], dtype=int)
X = np.expand_dims(np.concatenate([HC_weighted, SCH_weighted], axis=0), axis=1)
y = np.concatenate([HC_label, SCH_label], axis=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


train_data = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
test_data = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))

train_loader = DataLoader(train_data, batch_size=64)
val_loader = DataLoader(test_data, batch_size=64)
model = BrainNetCNN().to('mps')
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(6):
    print(epoch)
    model.train()
    train_preds, train_labels = [], []
    for batch_idx, (data, targets) in enumerate(train_loader):
        data, targets = data.to('mps'), targets.to('mps')
        scores = model(data)
        loss = criterion(scores, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_preds.extend(scores.argmax(dim=1).cpu().numpy())
        train_labels.extend(targets.cpu().numpy())

    train_accuracy = accuracy_score(train_labels, train_preds)
    print('train_accuracy', train_accuracy)
    model.eval()
    test_preds, test_labels = [], []
    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(val_loader):
            data, targets = data.to('mps'), targets.to('mps')
            scores = model(data)
            test_preds.extend(scores.argmax(dim=1).cpu().numpy())
            test_labels.extend(targets.cpu().numpy())
    test_accuracy = accuracy_score(test_labels, test_preds)
    print('test_accuracy', test_accuracy)
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
for data, targets in val_loader:
    data = data.to('mps')
    y_pred = model(data)
    y_pred = y_pred.argmax(dim=1).cpu().numpy()
    f1 = f1_score(targets, y_pred)
    precision = precision_score(targets, y_pred)
    auc = roc_auc_score(targets, y_pred)
    accuracy = accuracy_score(targets, y_pred)
    recall_score = recall_score(targets, y_pred)
print('f1', f1)
print('precision', precision)
print('auc', auc)
print('accuracy', accuracy)
print('recall_score', recall_score)
# model.eval()
# test_preds, test_labels = [], []
# with torch.no_grad():
#     for batch_idx, (data, targets) in enumerate(val_loader):
#         data, targets = data.to('mps'), targets.to('mps')
#         scores = model(data)
#         test_preds.extend(scores.argmax(dim=1).cpu().numpy())
#         test_labels.extend(targets.cpu().numpy())
#
# test_accuracy = accuracy_score(test_labels, test_preds)
#
# print(
#     f'Epoch {epoch + 1}/{10}, Loss: {loss.item()}, Train Accuracy: {train_accuracy}, Test Accuracy: {test_accuracy}')

#
#
# model = None
# criterion = torch.nn.CrossEntropyLoss()
# kf = KFold(n_splits=n_folds, shuffle=True)
#
# scores = []
# PATCH_SIZE = 4
# PATCHES_PER_SIDE = 32 // PATCH_SIZE
# PATCHES = PATCHES_PER_SIDE * PATCHES_PER_SIDE
# PATCH_DIM = PATCH_SIZE * PATCH_SIZE
#
# for fold, (train_idx, val_idx) in enumerate(kf.split(train_data)):
#     train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
#     val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)
#     train_loader = DataLoader(train_data, batch_size=64, sampler=train_subsampler)
#     val_loader = DataLoader(train_data, batch_size=64, sampler=val_subsampler)
#     if model_type == 'Transformer':
#         model = torch.nn.Transformer()
#
#         optimizer_hubs = torch.optim.Adam(model.parameters(), lr=0.01)
#
#     # 训练模型
#     for e in range(1, Epoch):
#         train(model, train_loader, optimizer_hubs, criterion)
#         # 验证模型
#         train_acc = test(model, train_loader, train_idx)
#         val_acc = test(model, val_loader, val_idx)
#         print(f'Epoch: {e:03d}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')
#     scores.append(val_acc)
# # 输出准确率
# print(f"accuracy: {scores}")
# print(f"Average Accuracy: {sum(scores) / len(scores)}")
#
# test_loader = DataLoader(test_data, batch_size=64)
# test_acc = test(model, test_loader, idx=test_loader.dataset)
# print(f'Test Acc: {test_acc:.4f}')
#
# model_path = 'checkpoints/model/{0}/{1}'.format(dataset, ts)
# if not os.path.exists(model_path):
#     os.makedirs(model_path)
# torch.save(model, model_path+'/'+model_type+'.pth')
#
# #
# # class GDPModel(torch.nn.Module):
# #     def __init__(self, num_features=4, hidden_size=16, target_size=2):
# #         super().__init__()
# #         self.hidden_size = hidden_size
# #         self.num_features = num_features
# #         self.target_size = target_size
# #         self.convs = [GATConv(self.num_features, self.hidden_size),
# #                       GATConv(self.hidden_size, self.hidden_size)]
# #         self.linear = Linear(self.hidden_size, self.target_size)
# #     def forward(self, data):
# #         x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
# #         for conv in self.convs[:-1]:
# #             x = conv(x, edge_index) # adding edge features here!
# #             x = F.relu(x)
# #             x = F.dropout(x, training=self.training)
# #         x = self.convs[-1](x, edge_index) # edge features here as well
# #
# #         # batch = torch.zeros(data.x.shape[0], dtype=int)
# #         x = global_mean_pool(x, batch)
# #         x = self.linear(x)
# #
# #         # return F.relu(x)
# #         return F.log_softmax(x, dim=1)
#
#
# # 将标签添加到Data对象中
#
# #
# # # loader = DataLoader(data_list, batch_size=16, shuffle=True)
# # model = model()  # 假设Net是你的网络类
# # optimizer_hubs = torch.optim.Adam(model.parameters(), lr=0.001)
# # criterion = torch.nn.CrossEntropyLoss()
# # for epoch in range(100):
# #     correct = 0
# #     total = 0
# #     for batch in train_loader:
# #         optimizer_hubs.zero_grad()
# #         out = model(batch)
# #         loss = criterion(out, batch.y)
# #         loss.backward()
# #         optimizer_hubs.step()
# #         _, predicted = torch.max(out, 1)
# #         total += batch.y.size(0)
# #         correct += (predicted == batch.y).sum().item()
# #     accuracy = 100 * correct / total
# #     print(f'train accuracy: {accuracy}%')
# #
# # model.eval()
# # correct = 0
# # total = 0
# #
# # with torch.no_grad():
# #     for batch in test_loader:
# #         out = model(batch)
# #         _, predicted = torch.max(out, 1)
# #         total += batch.y.size(0)
# #         correct += (predicted == batch.y).sum().item()
# #
# # accuracy = 100 * correct / total
# # print(f'Test accuracy: {accuracy}%')
# # # plt.imshow(HC_data[12], cmap='RdBu', interpolation='nearest', vmin=-1, vmax=1)
# # # # plt.show()
#
#
#
# # import matplotlib.pyplot as plt
# # plt.subplot(121)
# # plt.imshow(HC[0], cmap='RdBu', interpolation='nearest', vmin=0, vmax=1)
# # plt.subplot(122)
# # plt.imshow(SCH[0], cmap='RdBu', interpolation='nearest', vmin=0, vmax=1)
# # plt.show()
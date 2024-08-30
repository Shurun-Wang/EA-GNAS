import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score


def train(model, train_loader, optimizer, criterion, device):
    model.train()
    train_loss = 0.0
    batch_count = 0
    for i, data in enumerate(train_loader):  # Iterate in batches over the training dataset.
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
        loss = criterion(out, data.y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        train_loss += loss.item()
        batch_count += 1
    return train_loss/batch_count


def test(model, loader, criterion, device):
    model.eval()
    test_loss = 0.0
    correct, total = 0, 0
    batch_count = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        data = data.to(device)
        output = model(data.x, data.edge_index, data.batch)
        for o, l in zip(output.argmax(dim=1), data.y):
            if o == l:
                correct += 1
            total += 1
        loss = criterion(output, data.y)
        test_loss += loss.item()
        batch_count += 1
    return test_loss/batch_count, correct/total


def test_metrics(model, loader):
    model.eval()
    pred_list = []
    true_list = []

    for data in loader:  # Iterate in batches over the training/test dataset.
        tmp, out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        pred_list.extend(pred)
        true_list.extend(data.y)
    y_true = np.array(true_list)
    y_pred = np.array(pred_list)
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    print(accuracy, recall, precision, f1, roc_auc)
    # print('Test accuracy:', accuracy)
    # print('Test recall:', recall)
    # print('Test precision:', precision)
    # print('Test f1:', f1)
    # print('Test roc_auc:', roc_auc)
    return fpr, tpr


def fine_train(model, train_loader, optimizer, criterion, device, e, iters, scheduler):
    model.train()
    train_loss = 0.0
    batch_count = 0
    for i, data in enumerate(train_loader):  # Iterate in batches over the training dataset.
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
        loss = criterion(out, data.y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        train_loss += loss.item()
        batch_count += 1
        scheduler.step(e + i / iters)
    return train_loss/batch_count


def cal_trainset(model, train_loader, device):
    model.eval()
    correct, total = 0, 0
    for data in train_loader:  # Iterate in batches over the training/test dataset.
        data = data.to(device)
        output = model(data.x, data.edge_index, data.batch)
        for o, l in zip(output.argmax(dim=1), data.y):
            if o == l:
                correct += 1
            total += 1
    return correct/total


def cal_testset(model, test_loader, device, best_acc, best_y):
    model.eval()
    for data in test_loader:  # Iterate in batches over the training/test dataset.
        data = data.to(device)
        y_pred = model(data.x, data.edge_index, data.batch)
        f1 = f1_score(data.y, y_pred.argmax(dim=1))
        precision = precision_score(data.y, y_pred.argmax(dim=1))
        auc = roc_auc_score(data.y, y_pred.argmax(dim=1))
        accuracy = accuracy_score(data.y, y_pred.argmax(dim=1))
        recall = recall_score(data.y, y_pred.argmax(dim=1))
        # metrics = {accuracy, precision, recall, f1, auc}
        print(accuracy, precision, recall, f1, auc)
        if accuracy > best_acc:
            best_acc = accuracy
            # best_metrics = metrics
            best_y = y_pred

    return best_y, best_acc


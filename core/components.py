import torch.optim as op
import torch_geometric.nn as nn
import torch


def optim_build(model, codes):
    opt = codes[0]
    learning_rate = codes[1]
    weight_decay = codes[2]
    if opt == 'Adam':
        return op.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif opt == 'Sgd':
        return op.SGD(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif opt == 'Rms':
        return op.RMSprop(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        raise NotImplementedError


def layer_build(code1, code2, input_feature):
    # code2 = int(code2)
    # input_feature = int(input_feature)
    if code1 == 'GCNConv':
        return nn.GCNConv(in_channels=input_feature, out_channels=code2), code2
    elif code1 == 'ChebConv':
        return nn.ChebConv(in_channels=input_feature, out_channels=code2, K=3), code2
    elif code1 == 'SAGEConv':
        return nn.SAGEConv(in_channels=input_feature, out_channels=code2), code2
    elif code1 == 'GATConv':
        return nn.GATConv(in_channels=input_feature, out_channels=code2, heads=4), int(code2*4)
    elif code1 == 'ResGatedGraphConv':
        return nn.ResGatedGraphConv(in_channels=input_feature, out_channels=code2), code2
    else:
        raise NotImplementedError


def batch_build(code, input_feature):
    if code == None:
        return torch.nn.Identity()
    elif code == 'BatchNorm':
        return nn.BatchNorm(input_feature)
    elif code == 'LayerNorm':
        return nn.LayerNorm(input_feature)
    elif code == 'InstanceNorm':
        return nn.InstanceNorm(input_feature)
    else:
        raise NotImplementedError

def acti_build(code):
    if code == None:
        return torch.nn.Identity()
    elif code == 'Sigmoid':
        return torch.nn.Sigmoid()
    elif code == 'Tanh':
        return torch.nn.Tanh()
    elif code == 'Relu':
        return torch.nn.ReLU()
    elif code == 'Elu':
        return torch.nn.ELU()
    elif code == 'leaky_relu':
        return torch.nn.LeakyReLU()
    else:
        raise NotImplementedError

def drop_build(code):
    return torch.nn.Dropout(code)

def aggr_build(code):
    if code == 'Sum':
        return nn.aggr.SumAggregation()
    elif code == 'Mean':
        return nn.aggr.MeanAggregation()
    elif code == 'Max':
        return nn.aggr.MaxAggregation()
    elif code == 'Var':
        return nn.aggr.VarAggregation()
    elif code == 'Std':
        return nn.aggr.StdAggregation()
    else:
        raise NotImplementedError

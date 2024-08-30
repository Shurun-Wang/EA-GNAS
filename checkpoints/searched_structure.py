import numpy as np


def one_structure(ioa_name):
    if ioa_name == 'ga':
        structure = ['SAGEConv', 256, 'InstanceNorm', 'Relu', 0.2, 'ChebConv', 16, 'BatchNorm', 'Relu', 0.5, 'Max', 'Adam', 0.01, 0.001]
    elif ioa_name == 'ssa':
        structure = ['GATConv', 64, 'LayerNorm', 'Relu', 0.3, 'GATConv', 4, 'InstanceNorm', None, 0, 'ResGatedGraphConv', 4, 'InstanceNorm', 'Relu', 0, 'Std', 'Rms', 0.01, 0.001]
    elif ioa_name == 'woa':
        structure = ['ChebConv', 128, 'LayerNorm', 'leaky_relu', 0.4, 'GATConv', 128, None, 'leaky_relu', 0.4, 'ResGatedGraphConv', 128, 'InstanceNorm', 'leaky_relu', 0.5, 'Std', 'Rms', 0.001, 1e-05]
    elif ioa_name == 'gwo':
        structure = ['ChebConv', 256, 'InstanceNorm', 'leaky_relu', 0.2, 'GCNConv', 8, 'InstanceNorm', None, 0, 'Max', 'Adam', 0.01, 0.001]
    elif ioa_name == 'pso':
        structure = ['ChebConv', 128, 'BatchNorm', 'Elu', 0.1, 'ResGatedGraphConv', 128, None, 'Elu', 0, 'ChebConv', 64, 'InstanceNorm', 'Relu', 0.2, 'Max', 'Adam', 0.001, 0.001]
    elif ioa_name == 'so':
        structure = ['ResGatedGraphConv', 64, 'BatchNorm', None, 0.2, 'ChebConv', 16, 'BatchNorm', 'Relu', 0.5, 'Max', 'Adam', 0.01, 0.001]
    else:
        raise NotImplementedError

    return structure


if __name__ == '__main__':
    fitness = np.load('ssa/layer3/mct0.3/fitness.npy')
    mfood = np.load('ssa/layer3/mct0.3/mfood.npy')
    structure = np.load('ssa/layer3/mct0.3/structure.npy')
    # print(fitness[:, -1])
    pass
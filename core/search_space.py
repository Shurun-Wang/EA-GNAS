import numpy as np
import random


class SearchSpace(object):
    def __init__(self, num_gnn_layers):
        self.num_gnn_layers = num_gnn_layers
        self.gnn_space = {'layer_type': ['GCNConv', 'ChebConv', 'SAGEConv', 'GATConv', 'ResGatedGraphConv'],
                          'hidde_num': [4, 8, 16, 32, 64, 128, 256],
                          'batch_type': [None, 'BatchNorm', 'LayerNorm', 'InstanceNorm'],
                          'activ_type': [None, 'Sigmoid', 'Tanh', 'Relu', 'Elu', 'leaky_relu'],
                          'drop_rate': [0, 0.1, 0.2, 0.3, 0.4, 0.5]}
        self.aggr_space = {'aggr_type': ['Sum', 'Mean', 'Max', 'Var', 'Std']}
        self.opt_space = {'opt': ['Adam', 'Sgd', 'Rms'],
                          'learning_rate': [1e-2, 1e-3, 1e-4],
                          'weight_decay': [1e-3, 1e-4, 1e-5]}

    def initial_instance(self):
        """
        for a one-layer GNN: code sequences (5+1+3);
        for a two-layer GNN: code sequences (5+5+1+3);
        for a three-layer GNN: code sequences (5+5+5+1+3);
        """
        instance_codes = []
        for i in range(self.num_gnn_layers):
            layer_type, hidde_num, batch_type, activ_type, drop_rate = \
                random.uniform(0, 5), random.uniform(0, 7), random.uniform(0, 4), random.uniform(0, 6), random.uniform(0, 6)
            instance_codes.append(layer_type), instance_codes.append(hidde_num), instance_codes.append(batch_type)
            instance_codes.append(activ_type), instance_codes.append(drop_rate)

        aggr_type, opt, learning_rate, weight_decay = \
            random.uniform(0, 5), random.uniform(0, 3), random.uniform(0, 3), random.uniform(0, 3)
        instance_codes.append(aggr_type), instance_codes.append(opt)
        instance_codes.append(learning_rate), instance_codes.append(weight_decay)
        return np.array(instance_codes)

    def bounds(self):
        lower_bounds = np.zeros(self.num_gnn_layers*5+4)
        upper_bounds = np.array([len(self.gnn_space['layer_type']), len(self.gnn_space['hidde_num']),
                                 len(self.gnn_space['batch_type']), len(self.gnn_space['activ_type']),
                                 len(self.gnn_space['drop_rate'])]*self.num_gnn_layers+
                                [len(self.aggr_space['aggr_type'])]+[len(self.opt_space['opt'])]+
                                [len(self.opt_space['learning_rate'])]+[len(self.opt_space['weight_decay'])])
        return lower_bounds, upper_bounds-0.0001

    def modify_codes(self, codes_batch):
        # shape of input is [batch, codes_number] e.g., [30, 13] for a two-layer GNN
        lower_bounds, upper_bounds = self.bounds()
        modified_batch = []
        for i in range(codes_batch.shape[0]):
            modified_batch.append([min(max(d, lower), upper) for d, lower, upper in zip(codes_batch[i], lower_bounds, upper_bounds)])
        modified_batch = np.stack(modified_batch, axis=0)
        return modified_batch

    def de_layer_type(self, value):
        num_layer_type = len(self.gnn_space['layer_type'])
        if np.floor(value) in range(num_layer_type):
            return self.gnn_space['layer_type'][int(np.floor(value))]

    def de_hidde_num(self, value):
        num_hidde_num = len(self.gnn_space['hidde_num'])
        if np.floor(value) in range(num_hidde_num):
            return self.gnn_space['hidde_num'][int(np.floor(value))]

    def de_batch_type(self, value):
        num_batch_type = len(self.gnn_space['batch_type'])
        if np.floor(value) in range(num_batch_type):
            return self.gnn_space['batch_type'][int(np.floor(value))]

    def de_activ_type(self, value):
        num_activ_type = len(self.gnn_space['activ_type'])
        if np.floor(value) in range(num_activ_type):
            return self.gnn_space['activ_type'][int(np.floor(value))]

    def de_drop_rate(self, value):
        num_drop_rate = len(self.gnn_space['drop_rate'])
        if np.floor(value) in range(num_drop_rate):
            return self.gnn_space['drop_rate'][int(np.floor(value))]

    def de_aggr_type(self, value):
        num_aggr_type = len(self.aggr_space['aggr_type'])
        if np.floor(value) in range(num_aggr_type):
            return self.aggr_space['aggr_type'][int(np.floor(value))]

    def de_opt(self, value):
        num_opt = len(self.opt_space['opt'])
        if np.floor(value) in range(num_opt):
            return self.opt_space['opt'][int(np.floor(value))]

    def de_learning_rate(self, value):
        num_learning_rate = len(self.opt_space['learning_rate'])
        if np.floor(value) in range(num_learning_rate):
            return self.opt_space['learning_rate'][int(np.floor(value))]

    def de_weight_decay(self, value):
        num_weight_decay = len(self.opt_space['weight_decay'])
        if np.floor(value) in range(num_weight_decay):
            return self.opt_space['weight_decay'][int(np.floor(value))]

    def decode2net(self, codes):
        architectures = []
        for num in range(self.num_gnn_layers):
            architectures.append(self.de_layer_type(codes[num*5]))
            architectures.append(self.de_hidde_num(codes[num*5+1]))
            architectures.append(self.de_batch_type(codes[num*5+2]))
            architectures.append(self.de_activ_type(codes[num*5+3]))
            architectures.append(self.de_drop_rate(codes[num*5+4]))
        architectures.append(self.de_aggr_type(codes[-4]))
        architectures.append(self.de_opt(codes[-3]))
        architectures.append(self.de_learning_rate(codes[-2]))
        architectures.append(self.de_weight_decay(codes[-1]))
        return architectures

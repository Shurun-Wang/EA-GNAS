from core.components import layer_build, batch_build, acti_build, drop_build, aggr_build
import torch
import torch.nn.functional as F


class GraphNet(torch.nn.Module):
    def __init__(self, num_gnn_layers, codes, input_feature, explainable=False):
        super(GraphNet, self).__init__()
        self.num_gnn_layers = num_gnn_layers
        self.explainable = explainable
        self.layers = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.acti = torch.nn.ModuleList()
        self.drop = torch.nn.ModuleList()
        self.aggr = torch.nn.ModuleList()
        for i in range(self.num_gnn_layers):
            layer, out_feature = layer_build(codes[i*5], codes[i*5+1], input_feature)
            input_feature = out_feature
            batch = batch_build(codes[i*5+2], input_feature)
            acti = acti_build(codes[i*5+3])
            drop = drop_build(codes[i*5+4])
            self.layers.append(layer)
            self.bns.append(batch)
            self.acti.append(acti)
            self.drop.append(drop)
        aggr = aggr_build(codes[-1])
        self.aggr.append(aggr)
        self.lin = torch.nn.Linear(input_feature, 2)
        self.final_conv_acts = None
        self.final_conv_grads = None

    def activations_hook(self, grad):
        self.final_conv_grads = grad

    def forward(self, x, edge_index, batch):
        output = x
        for i, (layer, bn, acti, drop) in enumerate(zip(self.layers, self.bns, self.acti, self.drop)):
            if self.explainable:
                if i == self.num_gnn_layers-1:
                    with torch.enable_grad():
                        self.final_conv_acts = layer(output, edge_index)
                    self.final_conv_acts.register_hook(self.activations_hook)
                    output = drop(acti(bn(self.final_conv_acts)))
                else:
                    output = drop(acti(bn(layer(output, edge_index))))
            else:
                output = drop(acti(bn(layer(output, edge_index))))
        for aggr in self.aggr:
            output = aggr(output, batch)
        output = self.lin(output)
        output = F.log_softmax(output, dim=1)
        return output

import io
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, Linear


class EMOGI(nn.Module):
    """EMOGI model. A GCN with 3D graph convolutions and weighted loss.

        This class implements the EMOGI model. It is derived from the GCN
        model but contains some different metrics for logging (AUPR and AUROC
        for binary classification settings), a weighted loss function for
        imbalanced class sizes (eg. more negatives than positives) and
        the support for 3D graph convolutions (third dimension is treated
        similarly to channels in rgb images).
    """

    def __init__(self, input_dim, output_dim, num_hidden_layers=2, dropout_rate=0.5, hidden_dims=[20, 40]):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # model params
        self.num_hidden_layers = num_hidden_layers
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate

        # add intermediate layers
        # self.layers = []
        self.layers = nn.Sequential()
        inp_dim = self.input_dim
        for l in range(self.num_hidden_layers):
            self.layers.append(GCNConv(inp_dim,
                                       self.hidden_dims[l]))
            inp_dim = self.hidden_dims[l]

        self.layers.append(GCNConv(self.hidden_dims[-1],
                                   self.output_dim))
        # self.lin = Linear(self.hidden_dims[-1], self.output_dim)

    def forward(self, x, edge_index, edge_weight=None):
        for layer in self.layers[:-1]:
        # for layer in self.layers:
            x = layer(x, edge_index, edge_weight)
            x = F.relu(x)
            if self.dropout_rate is not None:
                x = F.dropout(x, self.dropout_rate, training=self.training)
        x = self.layers[-1](x, edge_index, edge_weight)
        # x = self.lin(x)

        return x


# x = torch.rand(100, 48)
# edge_index = torch.stack([torch.randint(low=0, high=100, size=(200,)), torch.randint(low=0, high=100, size=(200,))])
# data = Data(x=x, edge_index=edge_index)
#
# model = EMOGI(input_dim=48, output_dim=1)
# pred = model(data.x, data.edge_index)
# print(pred)

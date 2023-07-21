import numpy as np
import pandas as pd
import time
import pickle


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear

import torch_geometric.transforms as T
from torch_geometric.nn import ChebConv
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import dropout_adj, negative_sampling, remove_self_loops, add_self_loops, dropout_edge

from sklearn import metrics


class MTGCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MTGCN, self).__init__()
        self.conv1 = ChebConv(in_channels, 300, K=2, normalization="sym")
        self.conv2 = ChebConv(300, 100, K=2, normalization="sym")
        self.conv3 = ChebConv(100, out_channels, K=2, normalization="sym")

        self.lin1 = Linear(in_channels, 100)
        self.lin2 = Linear(in_channels, 100)

        self.c1 = torch.nn.Parameter(torch.Tensor([0.5]))
        self.c2 = torch.nn.Parameter(torch.Tensor([0.5]))

    def forward(self, x, edge_index, E, pb):
        # edge_index, _ = dropout_adj(edge_index, p=0.5,
        #                             force_undirected=True,
        #                             num_nodes=x.size()[0],
        #                             training=self.training)
        edge_index, _ = dropout_edge(edge_index=edge_index, p=0.5, force_undirected=True, training=self.training)
        x0 = F.dropout(x, training=self.training)
        x = torch.relu(self.conv1(x0, edge_index))
        x = F.dropout(x, training=self.training)
        x1 = torch.relu(self.conv2(x, edge_index))

        x = x1 + torch.relu(self.lin1(x0))
        z = x1 + torch.relu(self.lin2(x0))

        # The loss of link prediction
        pos_loss = -torch.log(torch.sigmoid((z[E[0]] * z[E[1]]).sum(dim=1)) + 1e-15).mean()

        # neg_edge_index = negative_sampling(pb, 100, 504378)
        neg_edge_index = negative_sampling(pb)

        neg_loss = -torch.log(
            1 - torch.sigmoid((z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=1)) + 1e-15).mean()

        r_loss = pos_loss + neg_loss

        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)

        return x, r_loss, self.c1, self.c2


# x = torch.rand(100, 48)
# edge_index = torch.stack([torch.randint(low=0, high=100, size=(200,)), torch.randint(low=0, high=100, size=(200,))])
# data = Data(x=x, edge_index=edge_index)
#
# pb, _ = remove_self_loops(data.edge_index)
# pb, _ = add_self_loops(pb)
# E = data.edge_index
#
# model= MTGCN()
# x,_,_,_= model(data, E, pb)
# print(x)
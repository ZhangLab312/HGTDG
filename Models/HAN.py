import os

import torch
from torch import nn
import torch.nn.functional as F

from torch_geometric.nn import HANConv
from torch_geometric.data import Data, HeteroData
import torch_geometric.transforms as T


class HAN(nn.Module):
    def __init__(self, metadata, in_channels, out_channels, hidden_size=64, heads=4, num_layers=3):
        super(HAN, self).__init__()
        # self.hidden_sizes = [hidden_size, hidden_size, hidden_size]
        self.layers = nn.Sequential()
        for layer in range(num_layers):
            self.layers.append(HANConv(in_channels=in_channels, out_channels=hidden_size, heads=heads, dropout=0.2,
                                metadata=metadata))
            in_channels = hidden_size
        #
        # self.han_conv = HANConv(in_channels=in_channels, out_channels=hidden_size, heads=heads, dropout=0.2,
        #                         metadata=metadata)
        # self.han_conv2 = HANConv(in_channels=hidden_size, out_channels=hidden_size, heads=heads, dropout=0.2,
        #                          metadata=metadata)
        self.lin = nn.Linear(in_features=hidden_size, out_features=out_channels)

    def forward(self, x_dict, edge_index_dict):
        # x_dict['protein'] = (x_dict['protein'] - x_dict['protein'].min()) / (
        #             x_dict['protein'].max() - x_dict['protein'].min())
        # out = self.han_conv(x_dict, edge_index_dict)
        out = x_dict
        for layer in self.layers:
            out = layer(out, edge_index_dict)
        out = self.lin(out['gene'])
        return out


# x1 = torch.rand(100, 48)
# x2 = torch.rand(200, 48)
# edge_index1 = torch.stack([torch.randint(low=0, high=100, size=(200,)), torch.randint(low=0, high=100, size=(200,))])
# edge_index2 = torch.stack([torch.randint(low=0, high=100, size=(400,)), torch.randint(low=0, high=100, size=(400,))])
# edge_index3 = torch.stack([torch.randint(low=0, high=100, size=(600,)), torch.randint(low=0, high=100, size=(600,))])
# data = HeteroData()
# data['gene'].x = x1
# data['x2'].x = x2
# data['gene', 'to', 'gene'].edge_index = edge_index1
# data['x2', 'to', 'x2'].edge_index = edge_index2
# data['gene', 'to', 'x2'].edge_index = edge_index3
# data = T.ToUndirected()(data)
# # print(data)
# model = HAN(metadata=data.metadata(), in_channels=48, out_channels=1)
# pred = model(data.x_dict, data.edge_index_dict)
# print(pred)
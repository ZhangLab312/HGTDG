# import os
# import time
# import pickle
# import numpy as np
# import pandas as pd
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch_geometric.transforms as T
#
# from torch_geometric.nn import Node2Vec
# from torch_geometric.data import Data, DataLoader
# from torch_geometric.utils import dropout_adj, negative_sampling, remove_self_loops, add_self_loops
#
# # from Utils.HGDataset_Homogeneous import HGDataset
# from Utils.HGDataset_MODIG import HGDataset
#
# os.chdir("/private/xiongshuwen/CancerGene/workingspace/MyDriverGenes")
#
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
# # x = torch.rand(100, 48)
# # edge_index = torch.stack([torch.randint(low=0, high=100, size=(200,)), torch.randint(low=0, high=100, size=(200,))])
# # data = Data(x=x, edge_index=edge_index)
# dataset = HGDataset(root="/private/xiongshuwen/CancerGene/workingspace/MyDriverGenes")
# data = dataset[0]
#
# model = Node2Vec(data.edge_index_dict['gene', 'to', 'gene'], embedding_dim=16, walk_length=80,
#                  context_size=5, walks_per_node=10,
#                  num_negative_samples=1, p=1, q=1, sparse=True).to(device)
# # model = Node2Vec(data.edge_index_dict['protein', 'to', 'protein'], embedding_dim=16, walk_length=80,
# #                  context_size=5, walks_per_node=10,
# #                  num_negative_samples=1, p=1, q=1, sparse=True).to(device)
#
# loader = model.loader(batch_size=128, shuffle=True)
# optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.001)
#
#
# def train():
#     model.train()
#     total_loss = 0
#     for pos_rw, neg_rw in loader:
#         optimizer.zero_grad()
#         out = model()
#         loss = model.loss(pos_rw.to(device), neg_rw.to(device))
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#     return total_loss / len(loader)
#
#
# for epoch in range(1, 501):
#     loss = train()
#     print(loss)
#
# model.eval()
# str_fearures = model()
#
# torch.save(str_fearures, 'ProcessedData_Homogeneous/str_fearures.pkl')
#
import os
import time
import pickle
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T

from torch_geometric.nn import Node2Vec
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import dropout_adj, negative_sampling, remove_self_loops, add_self_loops

# from Utils.HGDataset_Homogeneous import HGDataset
from Utils.HGDataset_MODIG import HGDataset

os.chdir("/private/xiongshuwen/CancerGene/workingspace/MyDriverGenes")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# x = torch.rand(100, 48)
# edge_index = torch.stack([torch.randint(low=0, high=100, size=(200,)), torch.randint(low=0, high=100, size=(200,))])
# data = Data(x=x, edge_index=edge_index)
dataset = HGDataset(root="/private/xiongshuwen/CancerGene/workingspace/MyDriverGenes")
data = dataset[0]

model = Node2Vec(data.edge_index_dict['gene', 'to', 'gene'], embedding_dim=16, walk_length=80,
                 context_size=5, walks_per_node=10,
                 num_negative_samples=1, p=1, q=1, sparse=True).to(device)

loader = model.loader(batch_size=128, shuffle=True)
optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.001)


def train():
    model.train()
    total_loss = 0
    for pos_rw, neg_rw in loader:
        optimizer.zero_grad()
        # out = model()
        loss = model.loss(pos_rw.to(device), neg_rw.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


for epoch in range(1, 501):
    loss = train()
    print(loss)

model.eval()
str_fearures = model()
str_fearures = torch.cat([str_fearures, str_fearures[-2:]])
torch.save(str_fearures, 'ProcessedData_Homogeneous/str_fearures.pkl')

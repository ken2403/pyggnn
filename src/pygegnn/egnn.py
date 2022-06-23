import torch
import torch.nn as nn
from torch import Tensor

from pygegnn import DataKeys

class EGNN(nn.Module):
    def __init__(self):
        super().__init__()
    
    def calc_atomic_distance(data)->Tensor:
        edge_src, edge_dst = data[DataKeys.Edge_index][0], data[DataKeys.Edge_index][1]
        edge_vec = (data[DataKeys.Position][edge_dst] - data[DataKeys.Position][edge_src]
            + torch.einsum('ni,nij->nj', data[DataKeys.Edge_shift], data[DataKeys.Lattice]))
        return 

    def forward(data_batch)->Tensor:
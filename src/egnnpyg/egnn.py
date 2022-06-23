import torch
import torch.nn as nn
from torch import Tensor

from egnnpyg import DataKeys

class EGNN(nn.Module):
    def __init__(self):
        super().__init__()
    
    def calc_atomic_distance(data)->Tensor:
        edge_src, edge_dst = data['edge_index'][0], data['edge_index'][1]
        edge_vec = (data['pos'][edge_dst] - data['pos'][edge_src]
            + torch.einsum('ni,nij->nj', data['edge_shift'], data['lattice']))

    def forward(data_batch)->Tensor:
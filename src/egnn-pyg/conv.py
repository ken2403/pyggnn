from typing import Tuple, Union, Optional

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj

from .swish import Swish
from .base import Dense


__all__ = ["EGNNConv"]


# class EGNNEdgeConv(MessagePassing):
#     def __init__(
#         self,
#         channels: Union[int, Tuple[int, int]],
#         node_dim: int,
#         hidden_dim: Optional[float] = None,
#         beta: Optional[float] = None,
#         residual: bool = True,
#         batch_norm: bool = True,
#     ):
#         super().__init__()
#         self.channels = channels
#         self.node_dim = node_dim
#         self.hidden_dim = hidden_dim
#         self.beta = beta
#         self.residual = residual
#         self.batch_norm = batch_norm

#         if isinstance(channels, int):
#             channels = (channels, channels)
#         if hidden_dim is None:
#             hidden_dim = channels[1] * 2
#         if residual:
#             assert channels[0] == channels[1]

#         # updata function
#         self.layers = nn.ModuleList(
#             [
#                 Dense(sum(channels) + node_dim, hidden_dim, bias=True),
#                 Swish(beta),
#                 Dense(hidden_dim, channels[1], bias=True),
#                 Swish(beta),
#             ]
#         )
#         if batch_norm:
#             self.bn = nn.BatchNorm1d(channels[1])
#         else:
#             self.bn = nn.Identity()

#     def reset_parameters(self):
#         for layer in self.layers:
#             layer.reset_parameters()

#     def forward(
#         self,
#         edge: Tensor,
#         edge_index: Adj,
#         edge_attr: Tensor,
#     ) -> Tensor:

#         # propagate_type: (x: Tensor, edge_attr: OptTensor)
#         out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=None)
#         out = self.bn(out)
#         out = out + x if self.residual else out
#         return out

#     def message(
#         self,
#         x_i: Tensor,
#         edge_attr: Tensor,
#     ) -> Tensor:
#         z = torch.cat([x_i, edge_attr], dim=1)
#         for layer in self.layers:
#             z = layer(z)
#         return z

#     def aggregate(self, inputs: Tensor, **kwargs) -> Tensor:
#         # no aggregation is exerted for edge convolution
#         return inputs


class EGNNConv(MessagePassing):
    def __init__(
        self,
        node_dim: Union[int, Tuple[int, int]],
        edge_dim: int,
        edge_attr_dim: Optional[int] = None,
        node_hidden: Optional[int] = None,
        edge_hidden: Optional[int] = None,
        beta: Optional[float] = None,
        residual: bool = True,
        batch_norm: bool = True,
        aggr: Optional[str] = "add",
        **kwargs,
    ):
        super().__init__(aggr=aggr, **kwargs)
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.edge_attr_dim = edge_attr_dim
        self.node_hidden = node_hidden
        self.edge_hidden = edge_hidden
        self.beta = beta
        self.residual = residual
        self.batch_norm = batch_norm

        if isinstance(node_dim, int):
            node_dim = (node_dim, node_dim)
        if edge_attr_dim is None:
            edge_attr_dim = 0
        if node_hidden is None:
            node_hidden = node_dim[1] * 2
        if edge_hidden is None:
            edge_hidden = edge_dim * 2
        if residual:
            assert node_dim[0] == node_dim[1]

        # updata function
        self.edge_update = nn.ModuleList(
            [
                Dense(
                    node_dim * 2 + edge_dim + 1 + edge_attr_dim, edge_hidden, bias=True
                ),
                Swish(beta),
                Dense(edge_hidden, edge_dim, bias=True),
                Swish(beta),
            ]
        )
        self.node_update = nn.ModuleList(
            [
                Dense(node_dim + edge_dim, node_hidden, bias=True),
                Swish(beta),
                Dense(node_hidden, node_dim[1], bias=True),
            ]
        )
        # inferring the edge
        self.inf = Dense(edge_dim, 1, bias=True, activation=F.sigmoid)
        if batch_norm:
            self.bn = nn.BatchNorm1d(node_dim[1])
        else:
            self.bn = nn.Identity()

    def reset_parameters(self):
        for nu in self.node_update:
            nu.reset_parameters()
        for eu in self.edge_update:
            eu.reset_parameters()
        self.inf.reset_parameters()
        self.bn.reset_parameters()

    def forward(
        self,
        node: Tensor,
        dist: Tensor,
        edge_index: Adj,
        edge_attr: Optional[Tensor] = None,
    ) -> Tensor:
        # propagate_type:
        # (node: Tensor, dist: Tensor, edge_attr: Optional[Tensor])
        edge = self.propagate(
            edge_index,
            x=node,
            dist=dist,
            edge_attr=edge_attr,
            size=None,
        )
        node_new = torch.cat([node, edge], dim=-1)
        for nu in self.node_update:
            node_new = nu(node_new)
        node_new = self.bn(node_new)
        node_new = node_new + node if self.residual else node_new
        return node_new

    def message(
        self,
        x_i: Tensor,
        x_j: Tensor,
        dist: Tensor,
        edge_attr: Optional[Tensor],
    ) -> Tensor:
        # update edge
        if edge_attr is None:
            edge_new = torch.cat([x_i, x_j, dist], dim=-1)
        else:
            assert edge_attr.size[-1] == self.edge_attr_dim
            edge_new = torch.cat([x_i, x_j, dist, edge_attr], dim=-1)
        for eu in self.edge_update:
            edge_new = eu(edge_new)

        # get inferring weight
        edge_new = self.inf(edge_new) * edge_new

        return edge_new

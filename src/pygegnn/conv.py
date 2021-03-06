from typing import Tuple, Union, Optional

import torch
from torch import Tensor
import torch.nn as nn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj

from pygegnn.activation import Swish
from pygegnn.base import Dense


__all__ = ["EGNNConv"]


# class EGNNEdgeConv(MessagePassing):
#     def __init__(
#         self,
#         channels: Union[int, Tuple[int, int]],
#         x_dim: int,
#         hidden_dim: Optional[float] = None,
#         beta: Optional[float] = None,
#         residual: bool = True,
#         batch_norm: bool = True,
#     ):
#         super().__init__()
#         self.channels = channels
#         self.x_dim = x_dim
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
#                 Dense(sum(channels) + x_dim, hidden_dim, bias=True),
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
    """
    The block to calculate massage passing and update node embeddings.
    It is implemented in the manner of PyTorch Geometric.
    """

    def __init__(
        self,
        x_dim: Union[int, Tuple[int, int]],
        edge_dim: int,
        edge_attr_dim: Optional[int] = None,
        node_hidden: int = 128,
        edge_hidden: int = 128,
        beta: Optional[float] = None,
        residual: bool = True,
        batch_norm: bool = True,
        aggr: Optional[str] = "add",
        **kwargs,
    ):
        """
        Args:
            x_dim (int or Tuple[int, int]]): number of node dimnsion. if set to tuple
                object, the first one is input dim, and second one is output dim.
            edge_dim (int): number of edge dim.
            edge_attr_dim (int or `None`, optional): number of another edge
                attribute dim.
                Defaults to `None`.
            node_hidden (int, optional): dimension of node hidden layers.
                Defaults to `128`.
            edge_hidden (int, optional): dimension of edge hidden layers.
                Defaults to `128`.
            beta (float or None, optional): beta coeff. Defaults to `None`.
            residual (bool, optional): if set to `False`, no residual network is used.
                Defaults to `True`.
            batch_norm (bool, optional): if set to `False`, no batch normalization is
                used. Defaults to `True`.
            aggr (str, optional): aggregation method. Defaults to `"add"`.
        """
        super().__init__(aggr=aggr, **kwargs)
        self.x_dim = x_dim
        self.edge_dim = edge_dim
        self.edge_attr_dim = edge_attr_dim
        self.node_hidden = node_hidden
        self.edge_hidden = edge_hidden
        self.beta = beta
        self.residual = residual
        self.batch_norm = batch_norm

        if isinstance(x_dim, int):
            x_dim = (x_dim, x_dim)
        if edge_attr_dim is None:
            edge_attr_dim = 0
        if residual:
            assert x_dim[0] == x_dim[1]

        # updata function
        self.edge_func = nn.ModuleList(
            [
                Dense(
                    x_dim[0] * 2 + 1 + edge_attr_dim,
                    edge_hidden,
                    bias=True,
                ),
                Swish(beta),
                Dense(edge_hidden, edge_dim, bias=True),
                Swish(beta),
            ]
        )
        self.node_func = nn.ModuleList(
            [
                Dense(x_dim[0] + edge_dim, node_hidden, bias=True),
                Swish(beta),
                Dense(node_hidden, x_dim[1], bias=True),
            ]
        )
        # inferring the edge
        self.inf = Dense(
            edge_dim, 1, bias=True, activation=torch.sigmoid, activation_name="sigmoid"
        )
        if batch_norm:
            self.bn = nn.BatchNorm1d(x_dim[1])
        else:
            self.bn = nn.Identity()

    def reset_parameters(self):
        for ef in self.edge_func:
            ef.reset_parameters()
        for nf in self.node_func:
            nf.reset_parameters()
        self.inf.reset_parameters()
        self.bn.reset_parameters()

    def forward(
        self,
        x: Tensor,
        dist: Tensor,
        edge_index: Adj,
        edge_attr: Optional[Tensor] = None,
    ) -> Tensor:
        # propagate_type:
        # (x: Tensor, dist: Tensor, edge_attr: Optional[Tensor])
        edge = self.propagate(
            edge_index,
            x=x,
            dist=dist,
            edge_attr=edge_attr,
            size=None,
        )
        out = torch.cat([x, edge], dim=-1)
        for nf in self.node_func:
            out = nf(out)
        out = self.bn(out)
        out = out + x if self.residual else out
        return out

    def message(
        self,
        x_i: Tensor,
        x_j: Tensor,
        dist: Tensor,
        edge_attr: Optional[Tensor],
    ) -> Tensor:
        # update edge
        if edge_attr is None:
            edge_new = torch.cat([x_i, x_j, dist.unsqueeze(-1)], dim=-1)
        else:
            assert edge_attr.size[-1] == self.edge_attr_dim
            edge_new = torch.cat([x_i, x_j, dist.unsqueeze(-1), edge_attr], dim=-1)
        for ef in self.edge_func:
            edge_new = ef(edge_new)

        # get inferring weight
        edge_new = self.inf(edge_new) * edge_new

        return edge_new

from typing import Literal, Optional

from torch import Tensor
import torch.nn as nn
from torch_geometric.nn import global_add_pool, global_mean_pool

from pygegnn import DataKeys
from .base import Dense
from .swish import Swish

__all__ = ["Node2Property"]


class Node2Property(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = DataKeys.Hidden_layer,
        out_dim: int = 1,
        beta: Optional[float] = None,
        aggr: Literal["add", "mean"] = "add",
    ):
        aggregation = {"add": global_add_pool, "mean": global_mean_pool}
        super().__init__()
        assert aggr == "add" or aggr == "mean"
        self.aggr = aggr
        self.layers = nn.Sequential(
            Dense(in_dim, hidden_dim, bias=True),
            Swish(beta),
            Dense(hidden_dim, hidden_dim, bias=True),
        )
        self.aggregate = aggregation[aggr]
        self.predict = nn.Sequential(
            Dense(hidden_dim, hidden_dim, bias=True),
            Swish(beta),
            Dense(hidden_dim, out_dim, bias=False),
        )

    def forward(self, x: Tensor, batch: Optional[Tensor] = None) -> Tensor:
        out = self.layers(x)
        out = self.aggregate(out, batch=batch)
        return self.predict(out)

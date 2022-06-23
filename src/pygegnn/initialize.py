from typing import Optional

import torch
from torch import Tensor
import torch.nn as nn

__all__ = ["AtomicNum2Node", "Distance2GaussianEdge"]


class AtomicNum2Node(nn.Embedding):
    def __init__(
        self,
        embedding_dim: int,
        max_num: Optional[int] = None,
    ):
        if max_num is None:
            max_num = 100
        super().__init__(
            num_embeddings=max_num, embedding_dim=embedding_dim, padding_idx=0
        )

    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x)


def gaussian_filter(
    distances: Tensor,
    offsets: Tensor,
    widths: Tensor,
    centered: bool = False,
) -> Tensor:
    """
    Filtered interatomic distance values using Gaussian functions.

    Notes:
        reference:
        [1] https://github.com/atomistic-machine-learning/schnetpack
    """
    if centered:
        # if Gaussian functions are centered, use offsets to compute widths
        eta = 0.5 / torch.pow(offsets, 2)
        # if Gaussian functions are centered, no offset is subtracted
        diff = distances[:, None]

    else:
        # compute width of Gaussian functions (using an overlap of 1 STDDEV)
        eta = 0.5 / torch.pow(widths, 2)
        # Use advanced indexing to compute the individual components
        diff = distances[:, None] - offsets[None, :]

    # compute smear distance values
    filtered_distances = torch.exp(-eta * torch.pow(diff, 2))
    return filtered_distances


class Distance2GaussianEdge(nn.Module):
    def __init__(
        self,
        start: float = 0.0,
        stop: float = 6.0,
        n_dim: int = 20,
        centered: bool = False,
        trainable: bool = True,
    ):
        super().__init__()
        offset = torch.linspace(start=start, end=stop, steps=n_dim)
        width = torch.Tensor((offset[1] - offset[0]) * torch.ones_like(offset))
        self.centered = centered
        if trainable:
            self.width = nn.Parameter(width)
            self.offset = nn.Parameter(offset)
        else:
            self.register_buffer("width", width)
            self.register_buffer("offset", offset)

    def forward(self, distances: Tensor) -> Tensor:
        """
        Compute filtered distances with Gaussian filter.

        Args:
            distances (Tensor): interatomic distance values of (num_edge) shape.

        Returns:
            filtered_distances (Tensor): filtered distances of (num_edge x n_dim) shape.
        """
        return gaussian_filter(
            distances, offsets=self.offset, widths=self.width, centered=self.centered
        )

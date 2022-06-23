from .base import *
from .conv import *
from .egnn import *
from .initialize import *
from .out import *
from .swish import *

__all__ = ["DataKeys"]


class DataKeys:
    X = "x"
    Edge_index = "edge_index"
    Position = "pos"
    Lattice = "lattice"
    Edge_shift = "edge_shift"
    Energy = "energy"
    Forces = "forces"

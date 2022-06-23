from .base import *
from .conv import *
from .egnn import *
from .initialize import *
from .out import *
from .swish import *

__all__ = ["DataKeys"]


class DataKeys:
    # data key name
    X = "x"
    Edge_index = "edge_index"
    Batch = "batch"
    Ptr = "ptr"
    Position = "pos"
    Atomic_num = "atomic_num"
    Lattice = "lattice"
    Edge_shift = "edge_shift"
    Edge_attr = "edge_attr"
    Energy = "energy"
    Forces = "forces"
    # model settings
    Hidden_layer = 128

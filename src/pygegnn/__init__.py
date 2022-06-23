from pygegnn.base import *
from pygegnn.conv import *
from pygegnn.egnn import *
from pygegnn.initialize import *
from pygegnn.out import *
from pygegnn.swish import *
from pygegnn.egnn import EGNN

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

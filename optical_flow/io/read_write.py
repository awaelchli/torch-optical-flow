from pathlib import Path
from typing import Union

from torch import Tensor

from optical_flow.io.kitti import read_kitti, write_kitti
from optical_flow.io.middlebury import read_middleburry, write_middlebury
from optical_flow.io.pfm import read_pfm, write_pfm

FORMATS = ["kitti", "middlebury", "pfm"]


def read(file: Union[str, Path], fmt="middlebury") -> Tensor:
    """Reads the flow map from a file.

    Args:
        file:
        fmt:
    """
    if fmt == "kitti":
        flow = read_kitti(file)
    elif fmt == "middlebury":
        flow = read_middleburry(file)
    elif fmt == "pfm":
        flow = read_pfm(file)
    else:
        raise ValueError(f"Unknown format {fmt}.")
    return flow


def write(file: Union[str, Path], flow: Tensor, fmt="middlebury") -> None:
    """Write optical flow to file.

    Args:
        flow:
        file:
        fmt:
    """
    flow = flow.cpu()
    assert flow.ndim == 3
    assert flow.shape[0] == 2

    if fmt == "kitti":
        write_kitti(file, flow)
    elif fmt == "middlebury":
        write_middlebury(file, flow)
    elif fmt == "pfm":
        write_pfm(file, flow)
    else:
        raise ValueError(f"Unknown format {fmt}")

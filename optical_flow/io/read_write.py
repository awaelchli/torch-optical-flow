from pathlib import Path
from typing import Union

from torch import Tensor

from optical_flow.io.kitti import read_kitti, write_kitti
from optical_flow.io.middlebury import read_middleburry, write_middlebury
from optical_flow.io.pfm import read_pfm, write_pfm

FORMATS = ["kitti", "middlebury", "pfm"]


def read(file: Union[str, Path], fmt="middlebury") -> Tensor:
    """Read optical flow data from a file.

    Supported are several common formats used to store optical flow data:

    - KITTI
    - Middlebury
    - PFM

    Args:
        file: path to a file to read the contents from
        fmt: name of the format in which optical flow is stored

    Returns:
        Optical flow in a torch tensor of shape (2, H, W).

    Raises:
        ValueError: If the given format string is not among the supported choices: kitti, middlebury, pfm.
    """
    if fmt == "kitti":
        flow = read_kitti(file)
    elif fmt == "middlebury":
        flow = read_middleburry(file)
    elif fmt == "pfm":
        flow = read_pfm(file)
    else:
        raise ValueError(f"Unknown format: {fmt}.")
    return flow


def write(file: Union[str, Path], flow: Tensor, fmt="middlebury") -> None:
    """Write optical flow to a file.

    Supported are several common formats used to store optical flow data:

    - KITTI
    - Middlebury
    - PFM

    Args:
        flow: the optical flow array or tensor of shape (2, H, W)
        file: a file path to where the contents will be written
        fmt: name of the format in which optical flow is stored

    Raises:
        ValueError: If the given format string is not among the supported choices: kitti, middlebury, pfm.
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
        raise ValueError(f"Unknown format: {fmt}")

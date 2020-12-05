from pathlib import Path
from typing import Union

from torch import Tensor

from optical_flow.io.middlebury import read_middleburry, write_middlebury
from optical_flow.io.pfm import read_pfm, write_pfm

FORMATS = ["middlebury", "pfm"]


def read(file: Union[str, Path], format="middlebury") -> Tensor:
    """Reads the flow map from a file.

    Args:
        file:
        format:
    """
    if format == "middlebury":
        flow = read_middleburry(file)
    elif format == "pfm":
        flow = read_pfm(file)
    else:
        raise ValueError(f"Unknown format {format}.")
    return flow


def write(file: Union[str, Path], flow: Tensor, format="middlebury"):
    """Write optical flow to file.

    Args:
        flow:
        file:
        format:
    """
    flow = flow.cpu()
    assert flow.ndim == 3
    assert flow.shape[0] == 2

    if format == "middlebury":
        write_middlebury(file, flow)
    elif format == "pfm":
        write_pfm(file, flow)
    else:
        raise ValueError(f"Unknown format {format}")

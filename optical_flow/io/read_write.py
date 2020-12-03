from pathlib import Path
from typing import Union

from torch import Tensor

from optical_flow.io.middlebury import read_middleburry, write_middlebury

FORMATS = ["middlebury"]


def read(file: Union[str, Path], format="middlebury") -> Tensor:
    """Reads the flow map from a file.

    Args:
        file:
        format:
    """
    if format == "middlebury":
        flow = read_middleburry(file)
    else:
        raise ValueError(f"Unknown format {format}.")
    return flow


def write(flow: Tensor, file: Union[str, Path], format="middlebury"):
    """Write optical flow to file.

    Args:
        flow:
        file:
        format:
    """
    if format == "middlebury":
        write_middlebury(flow, file)
    else:
        raise ValueError(f"Unknown format {format}")

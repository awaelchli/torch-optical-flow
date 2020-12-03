from pathlib import Path
from typing import Union

import torch
from torch import Tensor
import numpy as np


def read_middleburry(file: Union[str, Path]) -> Tensor:
    """ Read .flo file in Middlebury format

    Warning:
        This will work on little-endian architectures (eg Intel x86) only!

    Code adapted from:
    # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy
    """

    #

    with open(file, "rb") as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            raise RuntimeError("Magic number incorrect. Invalid .flo file.")
        w = int(np.fromfile(f, np.int32, count=1))
        h = int(np.fromfile(f, np.int32, count=1))
        data = np.fromfile(f, np.float32, count=(2 * w * h))

    data = np.resize(data, (int(h), int(w), 2)).transpose((2, 0, 1))
    data = torch.tensor(data)
    return data


def read(file: Union[str, Path], format="middlebury") -> Tensor:
    if format == "middlebury":
        flow = read_middleburry(file)
    else:
        raise ValueError(f"Unknown format {format}.")
    return flow

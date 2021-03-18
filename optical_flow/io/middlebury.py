from pathlib import Path
from typing import Union

import numpy as np
import torch
from torch import Tensor

MAGIC_NUMBER = 202021.25


def read_middleburry(file: Union[str, Path]) -> Tensor:
    """Read .flo file in Middlebury format

    Warning:
        This will work on little-endian architectures (eg Intel x86) only!

    Code adapted from:
    # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy
    """
    with open(file, "rb") as f:
        magic = np.fromfile(f, np.float32, count=1)
        if magic != MAGIC_NUMBER:
            raise RuntimeError("Magic number incorrect. Invalid .flo file.")
        w = int(np.fromfile(f, np.int32, count=1))
        h = int(np.fromfile(f, np.int32, count=1))
        data = np.fromfile(f, np.float32, count=(2 * w * h))

    data = np.resize(data, (int(h), int(w), 2)).transpose((2, 0, 1))
    data = torch.tensor(data)
    return data


def write_middlebury(file: Union[str, Path], flow: Union[Tensor, np.ndarray]) -> None:
    """
    Write optical flow to file in Middlebury format.
    Original code by Deqing Sun, adapted from Daniel Scharstein.

    Args:
        file:
        flow
    """
    if isinstance(flow, Tensor):
        flow = flow.cpu().numpy()

    assert flow.ndim == 3
    assert flow.shape[0] == 2
    u, v = flow[0], flow[1]
    height, width = u.shape
    with open(file, "wb") as f:
        # write the header
        f.write(np.array([MAGIC_NUMBER], np.float32))
        np.array(width).astype(np.int32).tofile(f)
        np.array(height).astype(np.int32).tofile(f)
        # arrange into matrix form
        tmp = np.zeros((height, width * 2))
        tmp[:, np.arange(width) * 2] = u
        tmp[:, np.arange(width) * 2 + 1] = v
        tmp.astype(np.float32).tofile(f)

from pathlib import Path
from typing import Union

from torch import Tensor
import numpy as np

MAGIC_NUMBER = 202021.25


def write_middlebury(flow: Union[Tensor, np.ndarray], file: Union[str, Path]):
    """
    Write optical flow to file in Middlebury format.
    Original code by Deqing Sun, adapted from Daniel Scharstein.

    Args:
        flow:
        file:
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


def write(flow: Tensor, file: Union[str, Path], format="middlebury"):
    """ Write optical flow to file.

    Args:
        flow:
        file:
        format:
    """
    if format == "middlebury":
        write_middlebury(flow, file)
    else:
        raise ValueError(f"Unknown format {format}")


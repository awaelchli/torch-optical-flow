import re
import sys
from pathlib import Path
from typing import Union

import numpy as np
import torch
from torch import Tensor


def read_pfm(file: Union[str, Path]) -> Tensor:
    """ PFM format used by FlyingThings3D

    Code adapted from
    https://github.com/liruoteng/OpticalFlowToolkit/blob/master/lib/pfm.py

    """

    file = open(file, "rb")
    header = file.readline().rstrip()
    if header == b"Pf":
        raise RuntimeError(
            "PFM file contains single-channel data. Cannot decode flow data."
        )
    if header != b"PF":
        raise RuntimeError("Not a PFM file.")

    dim_match = re.match(rb"^(\d+)\s(\d+)\s$", file.readline())
    if not dim_match:
        raise RuntimeError("Malformed PFM header. Cannot read spatial dimensions.")

    width, height = map(int, dim_match.groups())
    scale = float(file.readline().rstrip())
    endian = "<" if scale < 0 else ">"
    data = np.fromfile(file, endian + "f")
    shape = (height, width, 3)
    data = np.reshape(data, shape)
    data = data[:, :, :2]
    data = np.flipud(data)
    data = data.transpose((2, 0, 1))
    data = torch.tensor(data.copy())
    return data


def write_pfm(file: Union[str, Path], flow: Union[Tensor, np.ndarray]):
    """ PFM format used by FlyingThings3D

    Code adapted from
    https://github.com/liruoteng/OpticalFlowToolkit/blob/master/lib/pfm.py
    """
    # flow: (2, H, W)

    if isinstance(flow, Tensor):
        flow = flow.cpu().numpy()

    _, h, w = flow.shape
    assert flow.dtype == np.float32

    flow = flow.transpose((1, 2, 0))
    flow = np.flipud(flow)
    flow = np.concatenate((flow, np.zeros_like(flow, shape=(h, w, 1))), -1)
    endian = flow.dtype.byteorder
    scale = -1 if endian == "<" or endian == "=" and sys.byteorder == "little" else 1

    with open(file, "wb") as f:
        f.write("PF\n".encode())
        f.write(f"{w:d} {h:d}\n".encode())
        f.write(f"{scale:f}\n".encode())
        flow.tofile(f)

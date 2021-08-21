# MIT License
#
# Copyright (c) 2019 LI RUOTENG
# Copyright (c) 2021 Adrian Wälchli
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import re
import sys
from pathlib import Path
from typing import Union

import numpy as np
import torch
from torch import Tensor


def read_pfm(file: Union[str, Path]) -> Tensor:
    """Read optical flow file in PFM format used by the datasets from Uni Freiburg [1].

    Args:
        file: path to a file to read the contents from

    Returns:
        Optical flow in a torch tensor of shape (2, H, W).

    Raises:
        RuntimeError: If the file contains single-channel data only, is not a PFM file, or has a malformed PFM header.

    Note:
        Code adapted from Ruoteng Li [2].

    References:
        [1] N. Mayer, E. Ilg, P. Häusser, P. Fischer, D. Cremers, A. Dosovitskiy, T. Brox,
            "A Large Dataset to Train Convolutional Networks for Disparity, Optical Flow, and Scene Flow Estimation",
            CVPR, 2016.

        [2] Ruoteng Li, "Optical Flow Toolkit", 2016. URL: https://github.com/liruoteng/OpticalFlowToolkit
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


def write_pfm(file: Union[str, Path], flow: Union[Tensor, np.ndarray]) -> None:
    """Write optical flow to a file in PFM format used by the datasets from Uni Freiburg [1].

    Args:
        file: a file path to where the contents will be written
        flow: the optical flow array or tensor of shape (2, H, W)

    References:
        [1] N. Mayer, E. Ilg, P. Häusser, P. Fischer, D. Cremers, A. Dosovitskiy, T. Brox,
            "A Large Dataset to Train Convolutional Networks for Disparity, Optical Flow, and Scene Flow Estimation",
            CVPR, 2016.

        [2] Ruoteng Li, "Optical Flow Toolkit", 2016. URL: https://github.com/liruoteng/OpticalFlowToolkit
    """
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

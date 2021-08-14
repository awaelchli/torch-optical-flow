# The MIT License
#
# Copyright (c) 2018-2020 Tom Runia
# Copyright (c) 2021 Adrian Wälchli
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

from typing import Optional

import numpy as np
import torch
from torch import Tensor


def flow2rgb_baker(flow: Tensor) -> Tensor:
    """Convert an optical flow field to a colored representation in form of an RGB image using the method
    by Baker et al. [1].

    Args:
        flow: the optical flow tensor of shape (B, 2, H, W)

    Returns:
        RGB image of shape (B, 3, H, W)

    Note:
        - Code adapted from Tom Runia [2].
        - Code follows the original C++ source code of Daniel Scharstein.
        - Code follows the MATLAB source code of Deqing Sun.

    References:
        [1] S. Baker, D. Scharstein, J. Lewis, S. Roth, M. J. Black, and R. Szeliski,
            "A Database and Evaluation Methodology for Optical Flow", ICCV, 2007.
            URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf
        [2] Tom Runia, "Flow-vis: Optical Flow Visualization", 2020, URL: https://pypi.org/project/flow-vis
    """
    b, _, h, w = flow.shape
    u, v = flow[:, 0], flow[:, 1]
    colorwheel = colorwheel_baker(device=flow.device)  # (55, 3)
    ncols = colorwheel.shape[0]
    a = torch.atan2(-v, -u) / np.pi  # (B, H, W)
    fk = (a + 1) / 2 * (ncols - 1)
    k0 = torch.floor(fk).long()
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    f = f.view(-1, 1).expand(-1, 3)

    col0 = colorwheel[k0.view(-1), :] / 255.0
    col1 = colorwheel[k1.view(-1), :] / 255.0
    col = (1 - f) * col0 + f * col1
    rad = torch.norm(flow, p=2, dim=1).view(-1, 1).expand(-1, 3)  # (BHW, 3)
    idx = rad <= 1
    col[idx] = 1 - rad[idx] * (1 - col[idx])
    col[~idx] = col[~idx] * 0.75  # out of range
    flow_image = torch.floor(255 * col) / 255
    # (BHW, 3) -> (3, BHW) -> (3, B, H, W) -> (B, 3, H, W)
    flow_image = flow_image.permute(1, 0).view(3, b, h, w).permute(1, 0, 2, 3)
    return flow_image


def colorwheel_baker(device: Optional[torch.device] = None) -> Tensor:
    """Generates a color wheel for optical flow visualization as presented in [1].

    Args:
        device: the torch device to create the colorwheel on

    Returns:
        The color wheel

    Note:
        - Code adapted from Tom Runia [2].
        - Code follows the original C++ source code of Daniel Scharstein.
        - Code follows the MATLAB source code of Deqing Sun.

    References:
        [1] S. Baker, D. Scharstein, J. Lewis, S. Roth, M. J. Black, and R. Szeliski,
            "A Database and Evaluation Methodology for Optical Flow", ICCV, 2007.
            URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf
        [2] Tom Runia, "Flow-vis: Optical Flow Visualization", 2020, URL: https://pypi.org/project/flow-vis
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = torch.zeros(ncols, 3, device=device)
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = torch.floor(255 * torch.arange(0, RY, device=device) / RY)
    col = col + RY
    # YG
    colorwheel[col : col + YG, 0] = 255 - torch.floor(
        255 * torch.arange(0, YG, device=device) / YG
    )
    colorwheel[col : col + YG, 1] = 255
    col = col + YG
    # GC
    colorwheel[col : col + GC, 1] = 255
    colorwheel[col : col + GC, 2] = torch.floor(
        255 * torch.arange(0, GC, device=device) / GC
    )
    col = col + GC
    # CB
    colorwheel[col : col + CB, 1] = 255 - torch.floor(
        255 * torch.arange(CB, device=device) / CB
    )
    colorwheel[col : col + CB, 2] = 255
    col = col + CB
    # BM
    colorwheel[col : col + BM, 2] = 255
    colorwheel[col : col + BM, 0] = torch.floor(
        255 * torch.arange(0, BM, device=device) / BM
    )
    col = col + BM
    # MR
    colorwheel[col : col + MR, 2] = 255 - torch.floor(
        255 * torch.arange(MR, device=device) / MR
    )
    colorwheel[col : col + MR, 0] = 255
    return colorwheel

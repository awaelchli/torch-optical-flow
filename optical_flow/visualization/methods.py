from typing import Optional

import numpy as np
import torch
from torch import Tensor

from optical_flow.visualization.utils import hsv_to_rgb

EPS = 1e-5


def colorwheel_baker(device: Optional[torch.device] = None):
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf
    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.
    Returns:
        np.ndarray: Color wheel
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


def flow2rgb_baker(uv: Tensor):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.
    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun
    Args:
        uv: Input horizontal flow of shape [H,W]
    Returns:
        Flow visualization image of shape [H,W,3]
    """
    # uv: (B, 2, H, W)
    b, _, h, w = uv.shape
    u, v = uv[:, 0], uv[:, 1]
    colorwheel = colorwheel_baker(device=uv.device)  # (55, 3)
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
    rad = torch.norm(uv, p=2, dim=1).view(-1, 1).expand(-1, 3)  # (BHW, 3)
    idx = rad <= 1
    col[idx] = 1 - rad[idx] * (1 - col[idx])
    col[~idx] = col[~idx] * 0.75  # out of range
    flow_image = torch.floor(255 * col) / 255
    # (BHW, 3) -> (3, BHW) -> (3, B, H, W) -> (B, 3, H, W)
    flow_image = flow_image.permute(1, 0).view(3, b, h, w).permute(1, 0, 2, 3)
    return flow_image


def flow2rgb_hsv(flow: Tensor, max_norm: float = 1.0) -> Tensor:
    """
    Map optical flow to color image.
    The color hue is determined by the angle to the X-axis and the norm of the flow determines the saturation.
    White represents zero optical flow.

    :param flow: A torch.Tensor or numpy.ndarray of shape (B, 2, H, W). The components flow[:, 0] and flow[:, 1] are
    the X- and Y-coordinates of the optical flow, respectively.
    :param max_norm: The maximum norm of optical flow to be clipped. Default: 1.
    The optical flows that have a norm greater than max_value will be clipped for visualization.

    :return: Tensor of shape (B, 3, H, W)
    """
    flow = flow.clone()
    flow[:, 1] *= -1
    dx, dy = flow[:, 0], flow[:, 1]
    angle = torch.atan2(dy, dx)
    angle = torch.where(angle < 0, 2 * np.pi + angle, angle)

    scale = torch.sqrt(dx ** 2 + dy ** 2) / max_norm
    print(scale)
    h = angle / (2 * np.pi)
    s = torch.clamp(scale, 0, 1)
    v = torch.ones_like(s)

    hsv = torch.stack((h, s, v), 1)
    rgb = hsv_to_rgb(hsv)
    return rgb

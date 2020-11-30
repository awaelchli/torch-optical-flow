from typing import Union

import numpy as np
import torch
from torch import Tensor

from optical_flow.visualization.methods import (
    flow2rgb_baker,
    flow2rgb_hsv,
)

EPS = 1e-5


def flow2rgb(
    flow: Union[Tensor, np.ndarray],
    method: str = "baker",
    clip: float = None,
    max_norm: float = None,
    invert_y: bool = False,
) -> Tensor:
    """
    Args:
        flow:
        method:
        clip:
        max_norm:
        invert_y: Default: True. By default the optical flow is expected to be in a coordinate system with the
            Y axis pointing downwards. For intuitive visualization, the Y-axis is inverted.

    Returns:

    """

    # flow: (B, 2, H, W)
    if isinstance(flow, np.ndarray):
        flow = torch.as_tensor(flow)
    if clip is not None:
        flow = torch.clip(flow, -clip, clip)
    if invert_y:
        flow = flow.clone()
        flow[:, 1] *= -1

    max_norm = max_norm or torch.max(torch.norm(flow.flatten(2), p=2, dim=1), dim=1)[0].view(flow.shape[0], 1, 1, 1)
    flow = flow / (max_norm + EPS)

    if method == "baker":
        rgb = flow2rgb_baker(flow)

    elif method == "hsv":
        # todo: invert y properly handle defaults
        rgb = flow2rgb_hsv(flow, max_norm)

    else:
        raise ValueError(f"Unknown method '{method}'.")

    return rgb

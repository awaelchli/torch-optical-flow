from typing import Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor

from optical_flow.visualization.methods import flow2rgb_baker, flow2rgb_hsv

EPS = 1e-5
METHODS = [
    "baker",
    "hsv",
]


def flow2rgb(
    flow: Union[Tensor, np.ndarray],
    method: str = "baker",
    clip: Optional[Union[float, Tuple[float, float]]] = None,
    max_norm: Optional[float] = None,
    invert_y: bool = False,
) -> Tensor:
    """
    Args:
        flow:
        method:
        clip:
        max_norm: The maximum norm of optical flow to be clipped. Default: 1.
            The optical flows that have a norm greater than max_value will be clipped for visualization.
        invert_y: Default: True. By default the optical flow is expected to be in a coordinate system with the
            Y axis pointing downwards. For intuitive visualization, the Y-axis is inverted.

    Returns:

    """

    # flow: (B, 2, H, W)

    if isinstance(flow, np.ndarray):
        flow = torch.as_tensor(flow)
    ndims = flow.ndimension()
    if ndims == 3:
        flow = flow.unsqueeze(0)
    if clip is not None:
        clip = (-clip, clip) if not isinstance(clip, tuple) else clip
        flow = torch.clip(flow, clip[0], clip[1])
    if invert_y:
        flow = flow.clone()
        flow[:, 1] *= -1
    if max_norm is None:
        norm = torch.norm(flow.flatten(2), p=2, dim=1)
        max_norm = torch.max(norm, dim=1)[0].view(flow.shape[0], 1, 1, 1)
    flow = flow / (max_norm + EPS)

    if method == "baker":
        rgb = flow2rgb_baker(flow)
    elif method == "hsv":
        rgb = flow2rgb_hsv(flow)
    else:
        raise ValueError(f"Unknown method '{method}'.")

    if ndims == 3:
        rgb = rgb.view(*rgb.shape[-3:])
    return rgb


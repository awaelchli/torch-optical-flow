from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from optical_flow.visualization.methods import flow2rgb_baker, flow2rgb_hsv, flow2rgb_meister

EPS = 1e-5
METHODS = [
    "baker",
    "hsv",
    "meister",
]


def flow2rgb(
    flow: Union[Tensor, np.ndarray],
    method: str = "baker",
    clip: Optional[Union[float, Tuple[float, float]]] = None,
    max_norm: Optional[float] = None,
    invert_y: bool = False,
) -> Tensor:
    """Convert an optical flow field to a colored representation in form of an RGB image.

    Supports several common visualization methods: baker, hsv, meister

    Args:
        flow: the optical flow tensor of shape (2, H, W) or (B, 2, H, W)
        method: the name of the visualization method, one of: baker, hsv, meister
        clip: value for clipping the optical flow values. Applies before normalization with `max_norm`.
        max_norm: the maximum norm of optical flow to be clipped.
            The optical flows that have a norm greater than `max_norm` will be clipped for visualization.
        invert_y: by default the optical flow is expected to be in a coordinate system with the
            Y-axis pointing downwards. For intuitive visualization, the Y-axis is inverted.

    Returns:
        RGB image of shape (3, H, W) or (B, 3, H, W)

    Raises:
        ValueError: If the given name for the visualization method is not among the supported choices: baker, hsv,
            baker.
    """
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
    elif method == "meister":
        rgb = flow2rgb_meister(flow)
    else:
        raise ValueError(f"Unknown method: '{method}'.")

    if ndims == 3:
        rgb = rgb.view(*rgb.shape[-3:])
    return rgb


def colorwheel(
    method: str = "baker",
    size: int = 256,
    file: Optional[Union[str, Path]] = None,
) -> Tensor:
    """Renders an RGB colorwheel image for a given optical flow visualization method.

    Supports several common visualization methods: baker, hsv, meister

    Args:
        method: the name of the visualization method, one of: baker, hsv, meister
        size: the desired height of the output image in pixels
        file: optional file path to save the image to

    Returns:
        Square RGB image tensor of shape (3, H, H) where H is the given size argument
    """
    h = w = size
    max_norm = size / 2
    dy, dx = torch.meshgrid(
        torch.linspace(-h / 2, h / 2, h), torch.linspace(-w / 2, w / 2, w)
    )
    flow = torch.stack((dx, dy))
    norm = torch.norm(flow, dim=0, keepdim=True)
    rgb = flow2rgb(flow, method=method, max_norm=max_norm, invert_y=True)
    mask = torch.le(norm, max_norm)
    # white background
    rgb = mask * rgb + ~mask * torch.ones_like(rgb)

    if file is not None:
        rgb_numpy = rgb.mul(255).permute(1, 2, 0).type(torch.uint8).numpy()
        im = Image.fromarray(rgb_numpy, "RGB")
        im.save(file)

    return rgb

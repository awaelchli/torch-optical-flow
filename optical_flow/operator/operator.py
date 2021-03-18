from typing import Tuple, Union, Optional

import torch
from torch import Tensor
from torch.nn import functional as F


def warp(
    frame: Tensor,
    flow: Tensor,
    mode: str = "bilinear",
    padding_mode: str = "border",
    align_corners: bool = False,
) -> Tensor:
    """ Inverse warping with optical flow. """
    # frame: (B, C, H, W)
    # flow: (B, 2, H, W)
    flow = flow.permute(0, 2, 3, 1)  # swap flow dimensions (as expected by grid_sample)
    grid = warp_grid(flow)
    warped = F.grid_sample(
        frame, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners
    )
    return warped


def warp_grid(flow: Tensor) -> Tensor:
    # flow: (B, H, W, 2)
    b, h, w, _ = flow.shape
    range_x = torch.linspace(-1.0, 1.0, w, device=flow.device)
    range_y = torch.linspace(-1.0, 1.0, h, device=flow.device)
    grid_y, grid_x = torch.meshgrid(range_y, range_x)

    # grid has shape (B, H, W, 2)
    grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).repeat(b, 1, 1, 1)
    grid = grid + flow
    return grid


def scale(flow: Tensor, factor: Union[float, Tuple[float, float]] = 1) -> Tensor:
    """Scales the optical flow by a constant in X- and Y-direction.
    It is assumed that the flow map has a shape (B, 2, H, W), where B is the batch size.
    The first column of this tensor is the X-component and will be multiplied with factor[0].
    The second column is the Y-component of the optical flow and will be multiplied with factor[1].
    """
    # flow: (B, 2, H, W)
    assert flow.size(1) == 2
    if isinstance(factor, (float, int)):
        factor = (factor, factor)
    assert len(factor) == 2
    b, _, h, w = flow.shape
    scale_w = torch.empty_like(flow[:, 0]).fill_(factor[0])
    scale_h = torch.empty_like(flow[:, 0]).fill_(factor[1])
    factor = torch.stack((scale_w, scale_h), dim=1)
    return flow * factor


def resize(
    flow: Tensor,
    size: Optional[Tuple[int, int]] = None,
    scale_factor: Optional[float] = None,
    mode: str = "bilinear",
) -> Tensor:
    """ Resizes the optical flow in spatial dimensions and also multiplies the flow vectors by the scale factor. """
    assert flow.size(1) == 2
    assert flow.ndimension() == 4
    b, _, h, w = flow.shape
    if scale_factor:
        size = (round(h * scale_factor), round(w * scale_factor))
    sy = size[0] / h
    sx = size[1] / w
    resized = F.interpolate(flow, size, mode=mode)
    resized = scale(resized, (sx, sy))
    return resized


def normalize(flow: Tensor) -> Tensor:
    """ Normalizes the flow vectors in the range [-h, h] and [-w, w] to the range [-1, 1]. """
    # flow: (B, 2, H, W)
    assert flow.size(1) == 2
    h, w = flow.shape[-2:]
    return scale(flow, (2.0 / max(w - 1, 1), 2.0 / max(h - 1, 1)))


def denormalize(flow: Tensor) -> Tensor:
    """Inverts the normalization of the flow vectors in range [-1, 1]
    and brings it back to the ranges [-h, h] and [-w, w] for x- and y-components respectively.
    """
    # flow: (B, 2, H, W)
    assert flow.size(1) == 2
    h, w = flow.shape[-2:]
    return scale(flow, (max(w - 1, 1) / 2, max(h - 1, 1) / 2))


def integrate(*flows: Tensor) -> Tensor:
    # flows: (B, 2, H, W)
    assert len(flows) >= 2
    total = flows[-1]
    for flow in reversed(flows[:-1]):
        assert flow.shape == total.shape, "All flows must have the same size."
        total = flow + warp(total, flow)
    return total

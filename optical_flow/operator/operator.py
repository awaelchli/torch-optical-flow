from typing import Optional, Tuple, Union

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
    """Inverse warping with optical flow.

    Args:
        frame: the image tensor of shape (B, C, H, W)
        flow: the optical flow tensor of shape (B, 2, H, W)
        mode: the name of the interpolation method, see :func:`~torch.nn.functional.grid_sample` for more information
        padding_mode: the padding mode, see :func:`~torch.nn.functional.grid_sample` for more information
        align_corners: whether to align the corners of the sampling grid, see :func:`~torch.nn.functional.grid_sample`
            for more information

    Returns:
        The warped image
    """
    flow = flow.permute(0, 2, 3, 1)  # swap flow dimensions (as expected by grid_sample)
    grid = warp_grid(flow)
    warped = F.grid_sample(
        frame, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners
    )
    return warped


def warp_grid(flow: Tensor) -> Tensor:
    """Creates a warping grid from a given optical flow map.

    The warping grid determines the coordinates of the source pixels from which to take the color when inverse warping.

    Args:
        flow: optical flow tensor of shape (B, H, W, 2). The flow values are expected to already be in normalized range,
            see :func:`normalize` for more information.

    Returns:
        The warping grid
    """
    b, h, w, _ = flow.shape
    range_x = torch.linspace(-1.0, 1.0, w, device=flow.device)
    range_y = torch.linspace(-1.0, 1.0, h, device=flow.device)
    grid_y, grid_x = torch.meshgrid(range_y, range_x)

    # grid has shape (B, H, W, 2)
    grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).repeat(b, 1, 1, 1)
    grid = grid + flow
    return grid


def scale(flow: Tensor, factor: Union[float, Tuple[float, float]] = 1.0) -> Tensor:
    """Scales the optical flow by a constant in X- and Y-direction.

    The first column of this tensor is the X-component and will be multiplied with factor[0].
    The second column is the Y-component of the optical flow and will be multiplied with factor[1].

    Args:
        flow: optical flow tensor of shape (B, 2, H, W)
        factor: the scaling factor, either a single value for both dimensions or a tuple with individual scales for
            the X- and Y components

    Returns:
        The the optical flow map scaled in magnitude, of the same shape as the input flow tensor.
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
    """Resizes the optical flow in spatial dimensions and also re-scales the optical flow to account for the change in
    spatial coordinates.

    Args:
        flow: the optical flow tensor of shape (B, 2, H, W)
        size: the new spatial dimensions to resize the optical flow map to. Mutually exclusive with the `scale_factor`
            argument. The scale factor will be determined based on the given dimensions.
        scale_factor: the scaling factor for the spatial resizing and for the re-scaling of the optical flow magnitude.
            Mutually exclusive with the `size` argument.
        mode: the name of the interpolation method, see :func:`~torch.nn.functional.interpolate` for more information

    Returns:
        The resized optical flow tensor of shape (B, 2, H', W') with H' and W' being the new spatial dimensions.
    """
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
    """Re-scales the optical flow vectors such that they correspond to motion on the normalized pixel coordinates
    in the range [-1, 1] x [-1, 1].

    Args:
        flow: the optical flow tensor of shape (B, 2, H, W)

    Returns:
        The optical flow tensor with flow vectors rescaled to the normalized pixel coordinate system.
    """
    # flow: (B, 2, H, W)
    assert flow.size(1) == 2
    h, w = flow.shape[-2:]
    return scale(flow, (2.0 / max(w - 1, 1), 2.0 / max(h - 1, 1)))


def denormalize(flow: Tensor) -> Tensor:
    """Re-scales the optical flow vectors such that they correspond to motion on the regular integer pixel coordinates
    in the range [-h, h] x [-w, w].

    Args:
        flow: the optical flow tensor of shape (B, 2, H, W)

    Returns:
        The optical flow tensor with flow vectors rescaled to the regular integer pixel coordinate system.
    """
    # flow: (B, 2, H, W)
    assert flow.size(1) == 2
    h, w = flow.shape[-2:]
    return scale(flow, (max(w - 1, 1) / 2, max(h - 1, 1) / 2))


def integrate(*flows: Tensor) -> Tensor:
    """Integrate a sequence of optical flow maps to a single optical flow map that describes the combined motion from
    the first to the last coordinate frame.

    Args:
        *flows: the sequence of optical flow maps, all with the same shape (B, 2, H, W)

    Returns:
        The integration of optical flow over the whole sequence
    """
    # flows: (B, 2, H, W)
    assert len(flows) >= 2
    total = flows[-1]
    for flow in reversed(flows[:-1]):
        assert flow.shape == total.shape, "All flows must have the same size."
        total = flow + warp(total, flow)
    return total

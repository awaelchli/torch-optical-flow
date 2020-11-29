from typing import Union

import numpy as np
import torch
from torch import Tensor

from optical_flow.visualization.methods import (
    colorwheel_baker,
    flow2rgb_baker,
    flow2rgb_hsv,
)

EPS = 1e-5


def flow2rgb(
    flow: Union[Tensor, np.ndarray],
    method: str = "baker",
    clip: float = None,
    max_norm: float = None,
    invert_y: bool = True,
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

    max_norm = max_norm or torch.max(torch.norm(flow, p=2, dim=1))
    flow = flow / (max_norm + EPS)

    if method == "baker":
        rgb = flow2rgb_baker(flow)

    elif method == "hsv":
        # todo: invert y properly handle defaults
        rgb = flow2rgb_hsv(flow, max_norm)

    else:
        raise ValueError(f"Unknown method '{method}'.")

    return rgb


def main():
    from flow_vis.flow_vis import flow_to_color, make_colorwheel

    dev = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")

    x = make_colorwheel()
    y = colorwheel_baker()
    assert np.allclose(x, y.numpy())

    a = torch.randn(4)
    b = torch.randn(4)
    x = torch.atan2(a, b)
    y = np.arctan2(a.numpy(), b.numpy())
    assert np.allclose(x.numpy(), y)

    uv = torch.rand(1, 2, 5, 6, device=dev) * 100
    uv = uv.repeat(2, 1, 1, 1)

    y = flow_to_color(uv[0].cpu().permute(1, 2, 0).numpy())
    x = flow2rgb(uv)
    assert 0 <= x.min() <= x.max() <= 1
    # print(x[0].max(), y.max())
    assert np.allclose(x[0].permute(1, 2, 0).cpu().numpy(), y / 255)
    assert np.allclose(x[1].permute(1, 2, 0).cpu().numpy(), y / 255)
    assert x.device == dev
    assert x.dtype == torch.float
    # print(x.dtype, y.dtype)


if __name__ == "__main__":
    main()

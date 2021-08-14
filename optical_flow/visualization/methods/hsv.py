import numpy as np
import torch
from torch import Tensor

from optical_flow.visualization.utils import hsv_to_rgb


def flow2rgb_hsv(flow: Tensor) -> Tensor:
    """Convert an optical flow field to a colored representation in form of an RGB image using the method by
    representing the flow vectors in HSV color space and then converting it to RGB.

    The color hue is determined by the angle to the X-axis and the norm of the flow determines the saturation.
    White represents zero optical flow.

    Args:
        flow: the optical flow tensor of shape (B, 2, H, W)

    Returns:
        RGB image of shape (B, 3, H, W)
    """
    flow = flow.clone()
    flow[:, 1] *= -1

    dx, dy = flow[:, 0], flow[:, 1]
    angle = torch.atan2(dy, dx)
    angle = torch.where(angle < 0, angle + (2 * np.pi), angle)
    scale = torch.sqrt(dx ** 2 + dy ** 2)

    h = angle / (2 * np.pi)
    s = torch.clamp(scale, 0, 1)
    v = torch.ones_like(s)

    hsv = torch.stack((h, s, v), 1)
    rgb = hsv_to_rgb(hsv)
    return rgb

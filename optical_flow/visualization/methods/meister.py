# MIT License
#
# Copyright (c) 2017 Simon Meister
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

import numpy as np
import torch

from optical_flow.visualization.utils import hsv_to_rgb


def flow2rgb_meister(flow):
    """Convert an optical flow field to a colored representation in form of an RGB image using the method
    by Meister et al. [1].

    Args:
        flow: the optical flow tensor of shape (B, 2, H, W)

    Returns:
        RGB image of shape (B, 3, H, W)

    References:
        [1] S. Meister, J. Hur, S. Roth, "UnFlow: Unsupervised Learning of Optical Flow with a Bidirectional
            Census Loss", AAAI, 2018. URL: https://github.com/simonmeister/UnFlow
    """
    n = 8
    flow_u, flow_v = flow[:, 0], flow[:, 1]
    mag = torch.sqrt(flow_u ** 2 + flow_v ** 2)
    angle = torch.atan2(flow_v, flow_u)
    max_flow = torch.max(flow.flatten(1), dim=-1)[0]
    max_flow = max_flow.view(-1, 1, 1)
    im_h = torch.remainder(angle / (2 * np.pi) + 1.0, 1.0)
    im_s = torch.clip(mag * n / max_flow, 0, 1)
    im_v = torch.clip(n - im_s, 0, 1)
    im_hsv = torch.stack([im_h, im_s, im_v], 1)
    im = hsv_to_rgb(im_hsv)
    return im

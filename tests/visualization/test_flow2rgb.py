from typing import Callable

import numpy as np
import pytest
import torch
from flow_vis.flow_vis import flow_to_color
from torch import Tensor

from optical_flow.visualization.flow2rgb import METHODS, flow2rgb


def _batched_numpy_function(fn: Callable, inp: Tensor, **kwargs):
    array = inp.permute(0, 2, 3, 1).cpu().numpy()
    out = np.stack([fn(array[i], **kwargs) for i in range(array.shape[0])])
    return torch.tensor(out, device=inp.device, dtype=torch.float).permute(0, 3, 1, 2)


@pytest.mark.parametrize(
    "device",
    [
        pytest.param(torch.device("cpu")),
        pytest.param(
            torch.device("cuda", 0),
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="requires GPU"
            ),
        ),
    ],
)
@pytest.mark.parametrize("clip_flow", [None, 1.0, 50.0])
def test_flow2rgb_baker_parity(device, clip_flow):
    """ Test parity with flow-vis library (Baker visualization) """
    flow = (torch.rand(4, 2, 5, 6, device=device) * 2 - 1) * 100
    expected = _batched_numpy_function(flow_to_color, flow, clip_flow=clip_flow) / 255
    # the reference implemenation only clips the positive flow values
    # hence we need to limit our function to the same range to test parity
    clip_flow = (0, clip_flow) if clip_flow is not None else clip_flow
    output = flow2rgb(flow, method="baker", clip=clip_flow)
    assert torch.allclose(output, expected)
    assert 0 <= output.min() <= output.max() <= 1
    assert output.device == device
    assert output.dtype == torch.float


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("shape", [[1, 2, 3, 4], [4, 2, 3, 4], [2, 3, 4]])
def test_flow2rgb_input_output_shape(method, shape):
    """ Test that the function accepts both batched- and non-batched input tensors. """
    flow = (torch.rand(*shape) * 2 - 1) * 100
    output = flow2rgb(flow, method=method)
    expected = list(shape)
    expected[-3] = 3  # rgb
    assert list(output.shape) == expected


@pytest.mark.parametrize("method", METHODS)
def test_flow2rgb_numpy_conversion(method):
    """ Test that the function accepts a numpy array and converts it to a tensor. """
    flow = np.random.uniform(-100, 100, size=(4, 2, 5, 5))
    output = flow2rgb(flow, method=method)
    assert isinstance(output, Tensor)
    assert list(output.shape) == [4, 3, 5, 5]


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("clip", [1.0, 50.0])
def test_flow2rgb_clip(method, clip):
    """ Test that flow values get clipped to the correct values. """
    flow = (torch.rand(4, 2, 5, 6) * 2 - 1) * 100
    flow_clipped = torch.clip(flow, -clip, clip)
    output0 = flow2rgb(flow, method=method, clip=clip)
    output1 = flow2rgb(flow_clipped, method=method, clip=None)
    assert torch.allclose(output0, output1)
    assert 0 <= output0.min() <= output0.max() <= 1


@pytest.mark.parametrize("method", METHODS)
def test_flow2rgb_invert_y(method):
    """ Test that the Y axis can be inverted. """
    flow = torch.rand(4, 2, 5, 6)
    flow_inverted = flow.clone()
    flow_inverted[:, 1] *= -1
    output = flow2rgb(flow, invert_y=False)
    output_inverted = flow2rgb(flow_inverted, invert_y=True)
    assert torch.allclose(output, output_inverted)


@pytest.mark.parametrize("method", METHODS)
def test_flow2rgb_max_norm(method):
    assert False

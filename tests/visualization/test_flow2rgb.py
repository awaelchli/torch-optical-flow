from typing import Callable

import pytest
import torch
import numpy as np
from flow_vis.flow_vis import flow_to_color
from torch import Tensor

from optical_flow.visualization.flow2rgb import flow2rgb


def _batched_numpy_function(fn: Callable, inp: Tensor, **kwargs):
    array = inp.permute(0, 2, 3, 1).cpu().numpy()
    out = np.stack([fn(array[i], **kwargs) for i in range(array.shape[0])])
    return torch.tensor(out, device=inp.device, dtype=torch.float).permute(0, 3, 1, 2)


@pytest.mark.parametrize("device", [
    pytest.param(torch.device("cpu")),
    pytest.param(
        torch.device("cuda", 0),
        marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
    ),
])
def test_flow2rgb_baker_parity(device):
    """ Test parity with flow-vis library (Baker visualization) """
    # dev = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    flow = torch.rand(4, 2, 5, 6, device=device) * 100
    expected = _batched_numpy_function(flow_to_color, flow) / 255
    output = flow2rgb(flow, invert_y=False)
    assert torch.allclose(output, expected)
    assert output.device == device


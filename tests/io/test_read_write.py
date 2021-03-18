from pathlib import Path

import pytest
import torch

from optical_flow.io.read_write import FORMATS, read, write


@pytest.mark.parametrize("fmt", FORMATS)
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
def test_read_write(tmpdir, fmt, device):
    flow = torch.rand(2, 5, 6, device=device) * 100
    filename = Path(tmpdir) / "test"
    if fmt == "kitti":
        filename = filename.with_suffix(".png")
    write(filename, flow, fmt=fmt)
    loaded_flow = read(filename, fmt=fmt)
    assert isinstance(loaded_flow, torch.Tensor)
    assert loaded_flow.dtype == torch.float32
    assert loaded_flow.shape == flow.shape
    assert loaded_flow.device == torch.device("cpu")
    atol = 1e-1 if fmt == "kitti" else 1e-8
    assert torch.allclose(flow.cpu(), loaded_flow, atol=atol)

import pytest
import torch

from optical_flow.io.read_write import FORMATS, read, write


@pytest.mark.parametrize("format", FORMATS)
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
def test_read_write(tmpdir, format, device):
    flow = torch.randn(2, 5, 6, device=device) * 100
    write(tmpdir / "test", flow, format=format)
    loaded_flow = read(tmpdir / "test", format=format)
    assert isinstance(loaded_flow, torch.Tensor)
    assert loaded_flow.dtype == torch.float32
    assert loaded_flow.shape == flow.shape
    assert loaded_flow.device == torch.device("cpu")
    assert torch.allclose(flow, loaded_flow)

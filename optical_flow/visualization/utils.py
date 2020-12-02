import torch
from torch import Tensor


def hsv_to_rgb(image: Tensor) -> Tensor:
    r"""
    Convert an HSV image to RGB.
    The image data is assumed to be in the range of (0, 1).
    The code was taken and adapted from Kornia.
    See: https://github.com/kornia/kornia

    Args:
        image: RGB image to be converted to HSV.

    Returns:
        HSV version of the image.
    """

    assert torch.is_tensor(image)

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(
            "Input size must have a shape of (*, 3, H, W). Got {}".format(image.shape)
        )

    h = image[..., 0, :, :]
    s = image[..., 1, :, :]
    v = image[..., 2, :, :]

    hi = torch.floor(h * 6) % 6
    f = ((h * 6) % 6) - hi
    one = torch.tensor(1.0, device=image.device)
    p = v * (one - s)
    q = v * (one - f * s)
    t = v * (one - (one - f) * s)

    hi = hi.long()
    indices = torch.stack([hi, hi + 6, hi + 12], dim=-3)
    out = torch.stack(
        (
            v,
            q,
            p,
            p,
            t,
            v,
            t,
            v,
            v,
            q,
            p,
            p,
            p,
            p,
            t,
            v,
            v,
            q,
        ),
        dim=-3,
    )
    out = torch.gather(out, -3, indices)
    return out

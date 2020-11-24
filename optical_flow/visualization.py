import torch
import numpy as np
from torch import Tensor

EPS = 1e-5


def hsv_to_rgb(image: Tensor) -> Tensor:
    r"""Convert an HSV image to RGB.
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

    hi = torch.floor(h * 6)
    f = h * 6 - hi
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)

    out = torch.stack([hi, hi, hi], dim=-3) % 6

    out[out == 0] = torch.stack((v, t, p), dim=-3)[out == 0]
    out[out == 1] = torch.stack((q, v, p), dim=-3)[out == 1]
    out[out == 2] = torch.stack((p, v, t), dim=-3)[out == 2]
    out[out == 3] = torch.stack((p, q, v), dim=-3)[out == 3]
    out[out == 4] = torch.stack((t, p, v), dim=-3)[out == 4]
    out[out == 5] = torch.stack((v, p, q), dim=-3)[out == 5]

    return out


#
# def flow2rgb(flow: Tensor, max_norm: float = 1.0, invert_y: bool = True) -> Tensor:
#     """
#     Map optical flow to color image.
#     The color hue is determined by the angle to the X-axis and the norm of the flow determines the saturation.
#     White represents zero optical flow.
#
#     :param flow: A torch.Tensor or numpy.ndarray of shape (B, 2, H, W). The components flow[:, 0] and flow[:, 1] are
#     the X- and Y-coordinates of the optical flow, respectively.
#     :param max_norm: The maximum norm of optical flow to be clipped. Default: 1.
#     The optical flows that have a norm greater than max_value will be clipped for visualization.
#     :param invert_y: Default: True. By default the optical flow is expected to be in a coordinate system with the
#     Y axis pointing downwards. For intuitive visualization, the Y-axis is inverted.
#     :return: Tensor of shape (B, 3, H, W)
#     """
#     flow = flow.clone()
#     # flow: (B, 2, H, W)
#     if isinstance(flow, np.ndarray):
#         flow = torch.as_tensor(flow)
#     assert isinstance(flow, torch.Tensor)
#     assert max_norm > 0
#
#     if invert_y:
#         flow[:, 1] *= -1
#
#     dx, dy = flow[:, 0], flow[:, 1]
#
#     angle = torch.atan2(dy, dx)
#     angle = torch.where(angle < 0, 2 * np.pi + angle, angle)
#     scale = torch.sqrt(dx ** 2 + dy ** 2) / max_norm
#
#     h = angle / (2 * np.pi)
#     s = torch.clamp(scale, 0, 1)
#     v = torch.ones_like(s)
#
#     hsv = torch.stack((h, s, v), 1)
#     rgb = hsv_to_rgb(hsv)
#     return rgb


def make_colorwheel_torch():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf
    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.
    Returns:
        np.ndarray: Color wheel
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = torch.zeros(ncols, 3)
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = torch.floor(255 * torch.arange(0, RY) / RY)
    col = col + RY
    # YG
    colorwheel[col : col + YG, 0] = 255 - torch.floor(255 * torch.arange(0, YG) / YG)
    colorwheel[col : col + YG, 1] = 255
    col = col + YG
    # GC
    colorwheel[col : col + GC, 1] = 255
    colorwheel[col : col + GC, 2] = torch.floor(255 * torch.arange(0, GC) / GC)
    col = col + GC
    # CB
    colorwheel[col : col + CB, 1] = 255 - torch.floor(255 * torch.arange(CB) / CB)
    colorwheel[col : col + CB, 2] = 255
    col = col + CB
    # BM
    colorwheel[col : col + BM, 2] = 255
    colorwheel[col : col + BM, 0] = torch.floor(255 * torch.arange(0, BM) / BM)
    col = col + BM
    # MR
    colorwheel[col : col + MR, 2] = 255 - torch.floor(255 * torch.arange(MR) / MR)
    colorwheel[col : col + MR, 0] = 255
    return colorwheel


#
#
# def make_colorwheel_numpy():
#     """
#     Generates a color wheel for optical flow visualization as presented in:
#         Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
#         URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf
#     Code follows the original C++ source code of Daniel Scharstein.
#     Code follows the the Matlab source code of Deqing Sun.
#     Returns:
#         np.ndarray: Color wheel
#     """
#
#     RY = 15
#     YG = 6
#     GC = 4
#     CB = 11
#     BM = 13
#     MR = 6
#
#     ncols = RY + YG + GC + CB + BM + MR
#     colorwheel = np.zeros((ncols, 3))
#     col = 0
#
#     # RY
#     colorwheel[0:RY, 0] = 255
#     colorwheel[0:RY, 1] = np.floor(255 * np.arange(0, RY) / RY)
#     col = col + RY
#     # YG
#     colorwheel[col : col + YG, 0] = 255 - np.floor(255 * np.arange(0, YG) / YG)
#     colorwheel[col : col + YG, 1] = 255
#     col = col + YG
#     # GC
#     colorwheel[col : col + GC, 1] = 255
#     colorwheel[col : col + GC, 2] = np.floor(255 * np.arange(0, GC) / GC)
#     col = col + GC
#     # CB
#     colorwheel[col : col + CB, 1] = 255 - np.floor(255 * np.arange(CB) / CB)
#     colorwheel[col : col + CB, 2] = 255
#     col = col + CB
#     # BM
#     colorwheel[col : col + BM, 2] = 255
#     colorwheel[col : col + BM, 0] = np.floor(255 * np.arange(0, BM) / BM)
#     col = col + BM
#     # MR
#     colorwheel[col : col + MR, 2] = 255 - np.floor(255 * np.arange(MR) / MR)
#     colorwheel[col : col + MR, 0] = 255
#     return colorwheel

#
# def flow_uv_to_colors_numpy(u, v, convert_to_bgr=False):
#     """
#     Applies the flow color wheel to (possibly clipped) flow components u and v.
#     According to the C++ source code of Daniel Scharstein
#     According to the Matlab source code of Deqing Sun
#     Args:
#         u (np.ndarray): Input horizontal flow of shape [H,W]
#         v (np.ndarray): Input vertical flow of shape [H,W]
#         convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.
#     Returns:
#         np.ndarray: Flow visualization image of shape [H,W,3]
#     """
#     flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
#     colorwheel = make_colorwheel_numpy()  # shape [55x3]
#     ncols = colorwheel.shape[0]
#     rad = np.sqrt(np.square(u) + np.square(v))
#     a = np.arctan2(-v, -u) / np.pi
#     fk = (a + 1) / 2 * (ncols - 1)
#     k0 = np.floor(fk).astype(np.int32)
#     k1 = k0 + 1
#     k1[k1 == ncols] = 0
#     f = fk - k0
#     for i in range(colorwheel.shape[1]):
#         tmp = colorwheel[:, i]
#         col0 = tmp[k0] / 255.0
#         col1 = tmp[k1] / 255.0
#         print(f.shape, col0.shape, col1.shape)
#         col = (1 - f) * col0 + f * col1
#         idx = rad <= 1
#         col[idx] = 1 - rad[idx] * (1 - col[idx])
#         col[~idx] = col[~idx] * 0.75  # out of range
#         # Note the 2-i => BGR instead of RGB
#         ch_idx = 2 - i if convert_to_bgr else i
#         flow_image[:, :, ch_idx] = np.floor(255 * col)
#     return flow_image


def flow_uv_to_colors_torch(uv: Tensor):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.
    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun
    Args:
        u (np.ndarray): Input horizontal flow of shape [H,W]
        v (np.ndarray): Input vertical flow of shape [H,W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.
    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    # uv: (B, 2, H, W)
    b, _, h, w = uv.shape
    u, v = uv[:, 0], uv[:, 1]
    colorwheel = make_colorwheel_torch()  # (55, 3)
    ncols = colorwheel.shape[0]
    a = torch.atan2(-v, -u) / np.pi  # (B, H, W)
    fk = (a + 1) / 2 * (ncols - 1)
    k0 = torch.floor(fk).long()
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    f = f.view(-1, 1).expand(-1, 3)

    col0 = colorwheel[k0.view(-1), :] / 255.0
    col1 = colorwheel[k1.view(-1), :] / 255.0
    col = (1 - f) * col0 + f * col1
    rad = torch.norm(uv, p=2, dim=1).view(-1, 1).expand(-1, 3)  # (BHW, 3)
    idx = rad <= 1
    col[idx] = 1 - rad[idx] * (1 - col[idx])
    col[~idx] = col[~idx] * 0.75  # out of range
    flow_image = torch.floor(255 * col).type(torch.uint8)
    # (BHW, 3) -> (3, BHW) -> (3, B, H, W) -> (B, 3, H, W)
    flow_image = flow_image.permute(1, 0).view(3, b, h, w).permute(1, 0, 2, 3)
    return flow_image


# def flow_to_rgb(flow_uv: Tensor, clip_flow=None):
#     """
#     Expects a two dimensional flow image of shape.
#     Args:
#         flow_uv (np.ndarray): Flow UV image of shape [H,W,2]
#         clip_flow (float, optional): Clip maximum of flow values. Defaults to None.
#         convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.
#     Returns:
#         np.ndarray: Flow visualization image of shape [H,W,3]
#     """
#     assert flow_uv.ndim == 4, "input flow must have three dimensions"
#     assert flow_uv.shape[1] == 2, "input flow must have shape [H,W,2]"
#     if clip_flow is not None:
#         flow_uv = torch.clip(flow_uv, 0, clip_flow)
#     rad = torch.norm(flow_uv, p=2, dim=1)
#     rad_max = torch.max(rad)
#     epsilon = 1e-5
#     flow_uv = flow_uv / (rad_max + epsilon)
#     return flow_uv_to_colors_torch(flow_uv)


def flow2rgb(
    flow: Tensor,
    method: str = "baker",
    clip: float = None,
    max_norm: float = None,
    invert_y: bool = True,
) -> Tensor:
    if clip is not None:
        flow = torch.clip(flow, -clip, clip)
    max_norm = max_norm or torch.max(torch.norm(flow, p=2, dim=1))
    flow = flow / (max_norm + EPS)

    if method == "baker":
        rgb = flow_uv_to_colors_torch(flow)

    elif method == "hsv":
        # todo: invert y properly handle defaults
        rgb = None

    else:
        raise ValueError(f"Unknown method '{method}'.")

    return rgb


#
#
# def flow_to_color_numpy(flow_uv, clip_flow=None):
#     """
#     Expects a two dimensional flow image of shape.
#     Args:
#         flow_uv (np.ndarray): Flow UV image of shape [H,W,2]
#         clip_flow (float, optional): Clip maximum of flow values. Defaults to None.
#         convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.
#     Returns:
#         np.ndarray: Flow visualization image of shape [H,W,3]
#     """
#     assert flow_uv.ndim == 3, "input flow must have three dimensions"
#     assert flow_uv.shape[2] == 2, "input flow must have shape [H,W,2]"
#     if clip_flow is not None:
#         flow_uv = np.clip(flow_uv, 0, clip_flow)
#     u = flow_uv[:, :, 0]
#     v = flow_uv[:, :, 1]
#     rad = np.sqrt(np.square(u) + np.square(v))
#     rad_max = np.max(rad)
#     epsilon = 1e-5
#     u = u / (rad_max + epsilon)
#     v = v / (rad_max + epsilon)
#     return flow_uv_to_colors_numpy(u, v)


def main():

    from flow_vis.flow_vis import make_colorwheel, flow_to_color

    x = make_colorwheel()
    y = make_colorwheel_torch()
    assert np.allclose(x, y.numpy())

    a = torch.randn(4)
    b = torch.randn(4)
    x = torch.atan2(a, b)
    y = np.arctan2(a.numpy(), b.numpy())
    assert np.allclose(x.numpy(), y)

    uv = torch.rand(1, 2, 5, 6) * 100
    uv = uv.repeat(2, 1, 1, 1)

    y = flow_to_color(uv[0].permute(1, 2, 0).numpy())
    x = flow2rgb(uv)
    # print(x[0].max(), y.max())
    assert np.allclose(x[0].permute(1, 2, 0).numpy(), y)
    assert np.allclose(x[1].permute(1, 2, 0).numpy(), y)
    # print(x.dtype, y.dtype)


if __name__ == "__main__":
    main()

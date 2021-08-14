from pathlib import Path
from typing import Union, Tuple

import numpy as np
import torch
from torch import Tensor

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None


def _check_cv2_available():
    if not cv2:
        raise ModuleNotFoundError(
            "Reading and writing optical flow in KITTI format requires the opencv-python package."
            " To install it, run: pip install opencv-python-headless"
        )


def read_kitti(
    file: Union[str, Path], mask: bool = False
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Read optical flow file in KITTI [1] format.

    Args:
        file: path to a file to read the contents from
        mask: whether to return the mask for sparse flow

    Returns:
        Optical flow in a torch tensor of shape (2, H, W), and optionally a mask of shape (H, W) if `mask=True`.

    Raises:
        ModuleNotFoundError: If the opencv-python package is not installed.

    References:
        [1] Moritz Menze and Andreas Geiger, "Object Scene Flow for Autonomous Vehicles", CVPR, 2015.
    """
    _check_cv2_available()
    flow = cv2.imread(str(file), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    flow = flow[:, :, ::-1].astype(np.float32)
    flow, valid = flow[:, :, :2], flow[:, :, 2]
    flow = (flow - 2 ** 15) / 64.0
    flow = torch.tensor(flow).permute(2, 0, 1)
    valid = torch.tensor(valid)
    if mask:
        return flow, valid
    return flow


def write_kitti(file: Union[str, Path], flow: Union[Tensor, np.ndarray]) -> None:
    """Write optical flow to a file in KITTI [1] format.

    Args:
        file: a file path to where the contents will be written
        flow: the optical flow array or tensor of shape (2, H, W)

    Raises:
        ModuleNotFoundError: If the opencv-python package is not installed.

    References:
        [1] Moritz Menze and Andreas Geiger, "Object Scene Flow for Autonomous Vehicles", CVPR, 2015.
    """
    _check_cv2_available()
    if isinstance(flow, Tensor):
        flow = flow.cpu().numpy()
    flow = flow.transpose((1, 2, 0))
    flow = 64.0 * flow + 2 ** 15
    valid = np.ones([flow.shape[0], flow.shape[1], 1])
    flow = np.concatenate([flow, valid], axis=-1).astype(np.uint16)
    cv2.imwrite(str(file), flow[..., ::-1])

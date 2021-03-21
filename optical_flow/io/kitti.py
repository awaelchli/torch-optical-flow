from pathlib import Path
from typing import Tuple, Union

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
            " To install it, run: pip install opencv-python"
        )


def read_kitti(
    filename: Union[str, Path], return_mask: bool = False
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    _check_cv2_available()
    flow = cv2.imread(str(filename), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    flow = flow[:, :, ::-1].astype(np.float32)
    flow, valid = flow[:, :, :2], flow[:, :, 2]
    flow = (flow - 2 ** 15) / 64.0
    flow = torch.tensor(flow).permute(2, 0, 1)
    valid = torch.tensor(valid).unsqueeze(0)
    if return_mask:
        return flow, valid
    return flow


def write_kitti(filename: Union[str, Path], flow: Union[Tensor, np.ndarray]) -> None:
    _check_cv2_available()
    if isinstance(flow, Tensor):
        flow = flow.cpu().numpy()
    flow = flow.transpose((1, 2, 0))
    flow = 64.0 * flow + 2 ** 15
    valid = np.ones([flow.shape[0], flow.shape[1], 1])
    flow = np.concatenate([flow, valid], axis=-1).astype(np.uint16)
    cv2.imwrite(str(filename), flow[..., ::-1])

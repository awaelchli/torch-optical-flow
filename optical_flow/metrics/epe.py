from typing import Optional

import torch
from torch import Tensor
from torchmetrics import Metric


class AverageEndPointError(Metric):
    """Average End-to-end Point Error (AEPE or EPE for short).

    The end-to-end point error between two optical flow vectors is defined as the 2-norm of the residual vector.
    Supports passing in a mask to exclude invalid pixels.
    See also :func:`end_point_error` for a functional version of this metric.

    Args:
        dim: the dimension along which to compute the end-point-error
    """

    def __init__(self, dim: int = 1) -> None:
        super().__init__()
        self.dim = dim
        self.add_state("sum_epe", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(
        self, pred: Tensor, target: Tensor, valid: Optional[Tensor] = None
    ) -> None:
        epe = end_point_error(pred, target, dim=self.dim, reduce=False).view(-1)

        if valid is not None:
            valid = valid.view(-1) >= 0.5
            epe = epe[valid]

        self.sum_epe += epe.sum()
        self.total += epe.numel()

    def compute(self) -> Tensor:
        return self.sum_epe / self.total


def end_point_error(
    pred: Tensor, target: Tensor, dim: int = 1, reduce: bool = True
) -> Tensor:
    """End-to-end Point Error (EPE)

    The end-to-end point error between two optical flow vectors is defined as the 2-norm of the residual vector.

    Args:
        pred: predicted optical flow tensor of shape (B, 2, H, W)
        target: target optical flow tensor of shape (B, 2, H, W)
        dim: the dimension along which to compute the end-point-error
        reduce: whether to compute the mean over the pixel-wise end-point-errors

    Returns:
        Either a scalar tensor withthe average EPE (default) or a tensor of shape (B, H, W) with pixel-wise errors if
        the `reduce=False`.
    """
    epe = torch.norm(pred - target, p=2, dim=dim)
    if reduce:
        epe = epe.mean()
    return epe

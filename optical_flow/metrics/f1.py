from typing import Optional

import torch
from torch import Tensor

from optical_flow.metrics.epe import end_point_error
from torchmetrics import Metric


class OutlierRatio(Metric):
    """Outlier ratio, also known as F1.

    Measures the ratio of pixels that have an end-point-error greater than the given absolute threshold or
    have a relative error greater than the given relative threshold.
    """

    def __init__(
        self, dim: int = 1, abs_threshold: float = 3.0, rel_threshold: float = 0.05
    ) -> None:
        super().__init__()
        self.dim = dim
        self.abs_threshold = abs_threshold
        self.rel_threshold = rel_threshold
        self.add_state("sum_outliers", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(
        self, pred: Tensor, target: Tensor, valid: Optional[Tensor] = None
    ) -> None:
        epe = end_point_error(pred, target, dim=self.dim, reduce=False).view(-1)
        mag = torch.norm(target, p=2, dim=self.dim).view(-1)

        outliers = (
            (epe > self.abs_threshold) & ((epe / mag) > self.rel_threshold)
        ).float()

        if valid is not None:
            valid = valid.view(-1) >= 0.5
            outliers = outliers[valid]

        self.sum_outliers += outliers.sum()
        self.total += outliers.numel()

    def compute(self) -> Tensor:
        return self.sum_outliers / self.total

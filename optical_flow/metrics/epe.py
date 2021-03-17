import torch
from torchmetrics import Metric


class AverageEndPointError(Metric):
    """ Average End-to-end Point Error (AEPE) """

    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim
        self.add_state("sum_epe", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        epe = end_point_error(pred, target, dim=self.dim, reduce=False)
        self.sum_epe += epe.sum()
        self.total += epe.numel()

    def compute(self) -> torch.Tensor:
        return self.sum_epe / self.total


def end_point_error(
    pred: torch.Tensor, target: torch.Tensor, dim: int = 1, reduce: bool = True
):
    epe = torch.norm(pred - target, p=2, dim=dim)
    if reduce:
        epe = epe.mean()
    return epe

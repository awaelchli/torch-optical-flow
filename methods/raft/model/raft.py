from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.corr import CorrBlock
from model.extractor import BasicEncoder
from model.update import BasicUpdateBlock
from model.utils import coords_grid, InputPadder, upflow8
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import WandbLogger
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from wandb import Image

from optical_flow import flow2rgb
from optical_flow.metrics import AverageEndPointError


class RAFT(LightningModule):
    def __init__(
        self,
        hidden_dim: int = 128,
        context_dim: int = 128,
        corr_levels: int = 4,
        corr_radius: int = 4,
        iters: int = 12,
        gamma: float = 0.8,
        dropout: float = 0.0,
        lr: float = 0.00002,
        wdecay: float = 0.00005,
        epsilon: float = 1e-8,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # feature network, context network, and update block
        self.fnet = BasicEncoder(
            output_dim=256, norm_fn="instance", dropout=self.hparams.dropout
        )
        self.cnet = BasicEncoder(
            output_dim=(self.hparams.hidden_dim + self.hparams.context_dim),
            norm_fn="batch",
            dropout=self.hparams.dropout,
        )
        self.update_block = BasicUpdateBlock(
            self.hparams, hidden_dim=self.hparams.hidden_dim
        )

        # metrics
        self.epe_train = AverageEndPointError()
        self.epe_val = AverageEndPointError()

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    @staticmethod
    def initialize_flow(img):
        """Flow is represented as difference between two coordinate grids, flow = coords1 - coords0"""
        n, c, h, w = img.shape
        coords0 = coords_grid(n, h // 8, w // 8).to(img.device)
        coords1 = coords_grid(n, h // 8, w // 8).to(img.device)
        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    @staticmethod
    def upsample_flow(flow, mask):
        """Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination"""
        n, _, h, w = flow.shape
        mask = mask.view(n, 1, 9, 8, 8, h, w)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(n, 2, 9, 1, 1, h, w)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(n, 2, 8 * h, 8 * w)

    def forward(
        self, image0, image1, iters=12, flow_init=None, upsample=True, test_mode=False
    ):
        """Estimate optical flow between pair of frames"""

        image0 = 2 * (image0 / 255.0) - 1.0
        image1 = 2 * (image1 / 255.0) - 1.0

        image0 = image0.contiguous()
        image1 = image1.contiguous()

        hdim = self.hparams.hidden_dim
        cdim = self.hparams.context_dim

        # run the feature network
        fmap1, fmap2 = self.fnet([image0, image1])

        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
        corr_fn = CorrBlock(fmap1, fmap2, radius=self.hparams.corr_radius)

        # run the context network
        cnet = self.cnet(image0)
        net, inp = torch.split(cnet, [hdim, cdim], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(image0)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1)  # index correlation volume

            flow = coords1 - coords0
            net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)

            flow_predictions.append(flow_up)

        if test_mode:
            return coords1 - coords0, flow_up

        return flow_predictions

    def training_step(self, batch, batch_idx):
        img0, img1, flow, valid = batch
        flow_predictions = self(img0, img1, iters=self.hparams.iters)
        loss, metrics = sequence_loss(
            flow_predictions, flow, valid, gamma=self.hparams.gamma
        )
        self.epe_train(flow_predictions[-1], flow)

        self.log("loss", loss)
        self.log("epe_train", self.epe_train)
        self.log_dict(metrics)

        if self.global_step % 5000 == 0 and isinstance(self.logger, WandbLogger):
            self.logger.experiment.log(
                {
                    "images": [
                        Image(img0, caption="image 0"),
                        Image(img1, caption="image 1"),
                        Image(flow2rgb(flow), caption="GT flow"),
                        Image(flow2rgb(flow_predictions[-1]), caption="predicted flow"),
                    ],
                },
            )

        return loss

    def validation_step(self, batch, batch_idx):
        img0, img1, flow_gt, _ = batch

        padder = InputPadder(img0.shape)
        img0, img1 = padder.pad(img0, img1)
        _, flow_pr = self(img0, img1, iters=24, test_mode=True)
        flow_pr = padder.unpad(flow_pr)

        self.epe_val(flow_pr, flow_gt)
        self.log("epe_val", self.epe_val)

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.wdecay,
            eps=self.hparams.epsilon,
        )
        scheduler = OneCycleLR(
            optimizer,
            self.hparams.lr,
            self.trainer.max_steps + 100,
            pct_start=0.05,
            cycle_momentum=False,
            anneal_strategy="linear",
        )
        configuration = {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
        return configuration

    def on_validation_model_train(self):
        super().on_validation_model_train()
        if self.datamodule is not None and self.datamodule.stage != "chairs":
            self.freeze_bn()

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        # setting grads to None gives a modest performance improvement
        optimizer.zero_grad(set_to_none=True)


def sequence_loss(
    flow_preds: List[Tensor],
    flow_gt: Tensor,
    valid: Tensor,
    gamma: float = 0.8,
    max_flow: float = 400.0,
):
    """Loss function defined over sequence of flow predictions"""
    flow_preds = torch.stack(flow_preds)
    n_predictions = len(flow_preds)

    # exclude invalid pixels and extremely large displacements
    mag = torch.sum(flow_gt ** 2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < max_flow)

    idx = torch.arange(float(n_predictions), device=flow_gt.device)

    # the weight for each element in the sequence
    weight = gamma ** (n_predictions - idx - 1)

    losses = valid[None, :, None] * (flow_preds - flow_gt[None]).abs()
    flow_loss = torch.sum(weight * losses.flatten(1).mean(dim=1))

    epe = torch.sum((flow_preds[-1] - flow_gt) ** 2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        "1px": (epe < 1).float().mean().item(),
        "3px": (epe < 3).float().mean().item(),
        "5px": (epe < 5).float().mean().item(),
    }

    return flow_loss, metrics

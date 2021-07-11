import argparse
import os
import time
from argparse import ArgumentParser

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from data.datamodule import RAFTDataModule
from model import RAFT
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.utilities.cli import LightningCLI
from pytorch_lightning.core import datamodule
from torch.utils.data import DataLoader


class RAFTCLI(LightningCLI):

    def add_arguments_to_parser(self, parser):
        parser.add_argument("--name", default="raft", help="name your experiment")
        parser.add_argument("--restore_ckpt", help="restore checkpoint")
        parser.add_argument("--validation", type=str, nargs="+")

    def before_fit(self):
        if self.config["restore_ckpt"] is not None:
            self.model.load_state_dict(torch.load(self.config["restore_ckpt"]), strict=False)

        if self.datamodule.stage != "chairs":
            self.model.freeze_bn()


def main():
    seed_everything(1234)

    cli = RAFTCLI(
        RAFT,
        RAFTDataModule,
        description="RAFT",
        trainer_defaults=dict(
            max_steps=100000,
            gradient_clip_val=1.0,
            val_check_interval=5000,
        ),
    )

    #datamodule = RAFTDataModule(root_chairs="/Volumes/Archive/Datasets/FlyingChairs/data")


if __name__ == "__main__":
    main()

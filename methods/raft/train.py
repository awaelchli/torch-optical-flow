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
from pytorch_lightning.core import datamodule
from torch.utils.data import DataLoader


def main():
    seed_everything(1234)
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser.add_argument("--name", default="raft", help="name your experiment")
    parser.add_argument("--restore_ckpt", help="restore checkpoint")
    parser.add_argument("--validation", type=str, nargs="+")

    parser.set_defaults(
        max_steps=100000,
        gradient_clip_val=1.0,
        validate_every_n_steps=5000,
    )
    args = parser.parse_args()

    model = RAFT()
    datamodule = RAFTDataModule(root_chairs="/home/jovyan/optical-flow/FlyingChairs/data")

    if args.restore_ckpt is not None:
        model.load_state_dict(torch.load(args.restore_ckpt), strict=False)

    if datamodule.stage != "chairs":
        model.freeze_bn()

    trainer = Trainer.from_argparse_args(args)
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()

import argparse
from data.datamodule import RAFTDataModule
import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from pytorch_lightning.core import datamodule

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader

from argparse import ArgumentParser
from pytorch_lightning import Trainer, seed_everything
from model import RAFT


def main():
    seed_everything(1234)
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser.add_argument('--name', default='raft', help="name your experiment")
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--validation', type=str, nargs='+')

    parser.set_defaults(
        max_steps=100000,
        gradient_clip_val=1.0,
        validate_every_n_steps=5000,
    )
    args = parser.parse_args()
    
    model = RAFT()
    datamodule = RAFTDataModule()

    if args.restore_ckpt is not None:
        model.load_state_dict(torch.load(args.restore_ckpt), strict=False)

    if datamodule.stage != 'chairs':
        model.freeze_bn()

    trainer = Trainer.from_argparse_args(args)
    # trainer.fit(model)

if __name__ == '__main__':
    main()

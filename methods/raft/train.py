import argparse
import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

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
    parser.add_argument('--stage', help="determines which dataset to use for training") 
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--validation', type=str, nargs='+')
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--image_size', type=int, nargs='+', default=[384, 512])

    parser.set_defaults(
        max_steps=100000,
        gradient_clip_val=1.0,
        validate_every_n_steps=5000,
    )
    args = parser.parse_args()

    
    model = RAFT()
    if args.restore_ckpt is not None:
        model.load_state_dict(torch.load(args.restore_ckpt), strict=False)

    if args.stage != 'chairs':
        model.freeze_bn()

    trainer = Trainer.from_argparse_args(args)
    # trainer.fit(model)

if __name__ == '__main__':
    main()

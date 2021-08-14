import random

import numpy.random
import pytest
import torch

from methods.raft.data.dataset import MpiSintel, FlyingChairs, FlyingThings3D, KITTI
from methods.raft.data.dataset_new import MpiSintel as MpiSintelNew
from methods.raft.data.dataset_new import FlyingChairs as FlyingChairsNew
from methods.raft.data.dataset_new import FlyingThings3D as FlyingThings3DNew
from methods.raft.data.dataset_new import KITTI as KITTINew
import cv2

import numpy as np


ROOT_SINTEL = "/Volumes/Archive/Datasets/MPI-Sintel"
ROOT_CHAIRS = "/Volumes/Archive/Datasets/FlyingChairs/data"
ROOT_THINGS = "/Volumes/Archive/Datasets/FlyingThings3D"
ROOT_KITTI = "/Volumes/Archive/Datasets/KITTI-Flow-2015"


@pytest.mark.parametrize("split", ["training", "test"])
@pytest.mark.parametrize("dstype", ["final", "clean"])
@pytest.mark.parametrize(
    "aug_params",
    [
        None,
        {
            "crop_size": (368, 768),
            "min_scale": -0.5,
            "max_scale": 0.2,
            "do_flip": True,
        },
    ],
)
def test_sintel_new_old(split, dstype, aug_params):
    dataset_old = MpiSintel(
        split=split, dstype=dstype, root=ROOT_SINTEL, aug_params=aug_params
    )
    dataset_new = MpiSintelNew(
        split=split, dstype=dstype, root=ROOT_SINTEL, aug_params=aug_params
    )

    torch.manual_seed(100)
    numpy.random.seed(100)
    data_old = dataset_old[1]

    torch.manual_seed(100)
    numpy.random.seed(100)
    data_new = dataset_new[1]

    for old, new in zip(data_old, data_new):
        assert type(old) == type(new)
        if isinstance(old, torch.Tensor):
            assert torch.equal(old, new)


@pytest.mark.parametrize("split", ["training", "validation"])
@pytest.mark.parametrize(
    "aug_params",
    [
        None,
        {
            "crop_size": (368, 496),
            "min_scale": -0.1,
            "max_scale": 1.0,
            "do_flip": True,
        },
    ],
)
def test_flyingchairs_new_old(split, aug_params):
    dataset_old = FlyingChairs(aug_params=aug_params, split=split, root=ROOT_CHAIRS)
    dataset_new = FlyingChairsNew(aug_params=aug_params, split=split, root=ROOT_CHAIRS)

    torch.manual_seed(100)
    numpy.random.seed(100)
    data_old = dataset_old[1]

    torch.manual_seed(100)
    numpy.random.seed(100)
    data_new = dataset_new[1]

    for old, new in zip(data_old, data_new):
        assert type(old) == type(new)
        if isinstance(old, torch.Tensor):
            assert torch.equal(old, new)


@pytest.mark.parametrize("dstype", ["frames_cleanpass", "frames_cleanpass"])
@pytest.mark.parametrize(
    "aug_params",
    [
        None,
        {
            "crop_size": (400, 720),
            "min_scale": -0.4,
            "max_scale": 0.8,
            "do_flip": True,
        },
    ],
)
def test_flyingchairs3d_new_old(aug_params, dstype):
    dataset_old = FlyingThings3D(aug_params=aug_params, dstype=dstype, root=ROOT_THINGS)
    dataset_new = FlyingThings3DNew(
        aug_params=aug_params, dstype=dstype, root=ROOT_THINGS
    )

    torch.manual_seed(100)
    numpy.random.seed(100)
    data_old = dataset_old[1]

    torch.manual_seed(100)
    numpy.random.seed(100)
    data_new = dataset_new[1]

    for old, new in zip(data_old, data_new):
        assert type(old) == type(new)
        if isinstance(old, torch.Tensor):
            assert torch.equal(old, new)


@pytest.mark.parametrize(
    "split",
    [
        "training",
        "testing",
    ],
)
@pytest.mark.parametrize(
    "aug_params",
    [
        None,
        {
            "crop_size": (288, 960),
            "min_scale": -0.2,
            "max_scale": 0.4,
            "do_flip": False,
        },
    ],
)
def test_kitti_new_old(aug_params, split):
    dataset_old = KITTI(aug_params=aug_params, split=split, root=ROOT_KITTI)
    dataset_new = KITTINew(aug_params=aug_params, split=split, root=ROOT_KITTI)

    idx = random.randint(0, len(dataset_new))

    torch.manual_seed(100)
    numpy.random.seed(100)
    data_old = dataset_old[idx]

    torch.manual_seed(100)
    numpy.random.seed(100)
    data_new = dataset_new[idx]

    for old, new in zip(data_old, data_new):
        assert type(old) == type(new)
        if isinstance(old, torch.Tensor):
            assert torch.equal(old, new)

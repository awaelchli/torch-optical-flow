# BSD 3-Clause License
#
# Copyright (c) 2020, princeton-vl
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Data loading based on https://github.com/NVIDIA/flownet2-pytorch

import os
import random
from glob import glob

import numpy as np
import torch
from data.augmentor import FlowAugmentor, SparseFlowAugmentor
from PIL import Image
from torch.utils.data import Dataset

import optical_flow


class FlowDataset(Dataset):
    def __init__(self, aug_params=None, sparse=False):
        self.augmentor = None
        self.sparse = sparse
        if aug_params is not None:
            if sparse:
                self.augmentor = SparseFlowAugmentor(**aug_params)
            else:
                self.augmentor = FlowAugmentor(**aug_params)

        self.is_test = False
        self.init_seed = False
        self.flow_list = []
        self.image_list = []
        self.extra_info = []

    def __getitem__(self, index):

        if self.is_test:
            img1 = read_gen(self.image_list[index][0])
            img2 = read_gen(self.image_list[index][1])
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            return img1, img2, self.extra_info[index]

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)
        valid = None
        if self.sparse:
            flow, valid = optical_flow.read(
                self.flow_list[index], fmt="kitti", mask=True
            )
            valid = valid.numpy()
        else:
            flow = read_gen(self.flow_list[index])

        img1 = read_gen(self.image_list[index][0])
        img2 = read_gen(self.image_list[index][1])

        flow = flow.permute(1, 2, 0).numpy()
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)

        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[..., None], (1, 1, 3))
            img2 = np.tile(img2[..., None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        if self.augmentor is not None:
            if self.sparse:
                img1, img2, flow, valid = self.augmentor(img1, img2, flow, valid)
            else:
                img1, img2, flow = self.augmentor(img1, img2, flow)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()

        if valid is not None:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)

        return img1, img2, flow, valid.float()

    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        return self

    def __len__(self):
        return len(self.image_list)


class MpiSintel(FlowDataset):
    def __init__(
        self, aug_params=None, split="training", root="datasets/Sintel", dstype="clean"
    ):
        super(MpiSintel, self).__init__(aug_params)
        flow_root = os.path.join(root, split, "flow")
        image_root = os.path.join(root, split, dstype)

        if split == "test":
            self.is_test = True

        for scene in os.listdir(image_root):
            image_list = sorted(glob(os.path.join(image_root, scene, "*.png")))
            for i in range(len(image_list) - 1):
                self.image_list += [[image_list[i], image_list[i + 1]]]
                self.extra_info += [(scene, i)]  # scene and frame_id

            if split != "test":
                self.flow_list += sorted(glob(os.path.join(flow_root, scene, "*.flo")))


class FlyingChairs(FlowDataset):
    def __init__(
        self,
        aug_params=None,
        split="training",
        root="datasets/FlyingChairs_release/data",
    ):
        super(FlyingChairs, self).__init__(aug_params)

        images = sorted(glob(os.path.join(root, "*.ppm")))
        flows = sorted(glob(os.path.join(root, "*.flo")))
        assert len(images) // 2 == len(flows)

        split_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "chairs_split.txt"
        )
        split_list = np.loadtxt(split_file, dtype=np.int32)
        for i in range(len(flows)):
            xid = split_list[i]
            if (split == "training" and xid == 1) or (
                split == "validation" and xid == 2
            ):
                self.flow_list += [flows[i]]
                self.image_list += [[images[2 * i], images[2 * i + 1]]]


class FlyingThings3D(FlowDataset):
    def __init__(
        self, aug_params=None, root="datasets/FlyingThings3D", dstype="frames_cleanpass"
    ):
        super(FlyingThings3D, self).__init__(aug_params)

        for cam in ["left"]:
            for direction in ["into_future", "into_past"]:
                image_dirs = sorted(glob(os.path.join(root, dstype, "TRAIN/*/*")))
                image_dirs = sorted([os.path.join(f, cam) for f in image_dirs])

                flow_dirs = sorted(glob(os.path.join(root, "optical_flow/TRAIN/*/*")))
                flow_dirs = sorted([os.path.join(f, direction, cam) for f in flow_dirs])

                for idir, fdir in zip(image_dirs, flow_dirs):
                    images = sorted(glob(os.path.join(idir, "*.png")))
                    flows = sorted(glob(os.path.join(fdir, "*.pfm")))
                    for i in range(len(flows) - 1):
                        if direction == "into_future":
                            self.image_list += [[images[i], images[i + 1]]]
                            self.flow_list += [flows[i]]
                        elif direction == "into_past":
                            self.image_list += [[images[i + 1], images[i]]]
                            self.flow_list += [flows[i + 1]]


class KITTI(FlowDataset):
    def __init__(self, aug_params=None, split="training", root="datasets/KITTI"):
        super(KITTI, self).__init__(aug_params, sparse=True)
        if split == "testing":
            self.is_test = True

        root = os.path.join(root, split)
        images1 = sorted(glob(os.path.join(root, "image_2/*_10.png")))
        images2 = sorted(glob(os.path.join(root, "image_2/*_11.png")))

        for img1, img2 in zip(images1, images2):
            frame_id = img1.split("/")[-1]
            self.extra_info += [[frame_id]]
            self.image_list += [[img1, img2]]

        if split == "training":
            self.flow_list = sorted(glob(os.path.join(root, "flow_occ/*_10.png")))


class HD1K(FlowDataset):
    def __init__(self, aug_params=None, root="datasets/HD1k"):
        super(HD1K, self).__init__(aug_params, sparse=True)

        seq_ix = 0
        while 1:
            flows = sorted(
                glob(os.path.join(root, "hd1k_flow_gt", "flow_occ/%06d_*.png" % seq_ix))
            )
            images = sorted(
                glob(os.path.join(root, "hd1k_input", "image_2/%06d_*.png" % seq_ix))
            )

            if len(flows) == 0:
                break

            for i in range(len(flows) - 1):
                self.flow_list += [flows[i]]
                self.image_list += [[images[i], images[i + 1]]]

            seq_ix += 1


def read_gen(filename):
    ext = os.path.splitext(filename)[-1]
    if ext == ".png" or ext == ".jpeg" or ext == ".ppm" or ext == ".jpg":
        return Image.open(filename)
    elif ext == ".flo":
        return optical_flow.read(filename, fmt="middlebury")
    elif ext == ".pfm":
        return optical_flow.read(filename, fmt="pfm")
    return []

# Data loading based on https://github.com/NVIDIA/flownet2-pytorch
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset


class OpticalFlowDataset(Dataset, ABC):
    def __init__(self):
        super().__init__()
        self._has_init_seed = False

    @abstractmethod
    def image_files(self, index: int) -> Tuple:
        pass

    @abstractmethod
    def flow_file(self, index: int) -> Optional[Union[str, Path]]:
        pass

    def extra_info(self, index: int) -> Any:
        return index

    def read_image(self, filename: Union[str, Path]) -> Tensor:
        img = Image.open(filename)
        img = np.array(img).astype(np.uint8)[..., :3]
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        return img

    @abstractmethod
    def read_flow(self, filename: Union[str, Path]) -> Tuple[Tensor, Optional[Tensor]]:
        # # optical_flow.read(self.flow_list[index], fmt="kitti", return_mask=True)
        # valid = None
        # if self.sparse:
        #     flow, valid = frame_utils.readFlowKITTI(filename)
        # else:
        #     flow = frame_utils.read_gen(filename)
        #
        # return flow, valid
        pass

    def __getitem__(self, index: int):
        index = index % len(self)

        img1 = Image.open(self.image_files(index)[0])
        img2 = Image.open(self.image_files(index)[1])
        img1 = np.array(img1).astype(np.uint8)[..., :3]
        img2 = np.array(img2).astype(np.uint8)[..., :3]
        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()

        flow_file = self.flow_file(index)
        if flow_file is None:
            return img1, img2, self.extra_info(index)

        self._init_seed()

        flow, valid = self.read_flow(flow_file)

        # if self.augmentor is not None:
        #     if self.sparse:
        #         img1, img2, flow, valid = self.augmentor(img1, img2, flow, valid)
        #     else:
        #         img1, img2, flow = self.augmentor(img1, img2, flow)

        if valid is None:
            valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)
            valid = valid.unsqueeze(0)
        valid = valid.float()

        return img1, img2, flow, valid

    # def __rmul__(self, v: int):
    #     self._flow_list = v * self._flow_list
    #     self._image_list = v * self._image_list
    #     return self

    # def __len__(self):
    #     return len(self._image_list)

    def _init_seed(self):
        if self._has_init_seed:
            return
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            torch.manual_seed(worker_info.id)
            np.random.seed(worker_info.id)
            random.seed(worker_info.id)
            self._has_init_seed = True

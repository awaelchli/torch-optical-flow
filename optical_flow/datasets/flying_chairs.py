from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from torch import Tensor

import optical_flow
from optical_flow.datasets.base import OpticalFlowDataset

TRAIN_VAL_SPLIT_FILE = Path(__file__).parent / "chairs_split.txt"


class FlyingChairs(OpticalFlowDataset):
    def __init__(self, root: Union[str, Path], split="training"):
        super(FlyingChairs, self).__init__()
        self.split = split
        self._flow_list = sorted(Path(root).glob("*.flo"))
        self._image_list = []
        images = sorted(Path(root).glob("*.ppm"))
        assert len(images) // 2 == len(self._flow_list)

        split_list = np.loadtxt(TRAIN_VAL_SPLIT_FILE, dtype=np.int32)
        for i in range(len(self._flow_list)):
            xid = split_list[i]
            if (split == "training" and xid == 1) or (
                split == "validation" and xid == 2
            ):
                self._image_list += [(images[2 * i], images[2 * i + 1])]

    def image_files(self, index: int) -> Tuple:
        return self._image_list[index]

    def flow_file(self, index: int) -> Optional[Union[str, Path]]:
        return self._flow_list[index]

    def read_flow(self, filename: Union[str, Path]) -> Tuple[Tensor, None]:
        flow = optical_flow.read(filename, fmt="middlebury")
        valid = None
        return flow, valid

    def __len__(self):
        return len(self._image_list)

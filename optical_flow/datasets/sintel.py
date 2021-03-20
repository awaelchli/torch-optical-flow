import os
from pathlib import Path
from typing import Optional, Tuple, Union

from torch import Tensor

import optical_flow
from optical_flow.datasets.base import OpticalFlowDataset


class MPISintel(OpticalFlowDataset):
    def __init__(
        self, root: Union[str, Path], split: str = "training", dstype: str = "clean"
    ):
        super().__init__()
        self.root = root
        self.split = split
        self.dstype = dstype
        self.root_flow = Path(root, split, "flow")
        self.root_image = Path(root, split, dstype)
        self.scenes = os.listdir(self.root_image)
        self._image_list = []
        self._flow_list = []
        self._extra_info = []

        for scene in self.scenes:
            image_list = sorted(Path(self.root_image, scene).glob("*.png"))

            for i in range(len(image_list) - 1):
                self._image_list += [(image_list[i], image_list[i + 1])]
                self._extra_info += [(scene, i)]

            if split != "test":
                self._flow_list += sorted(Path(self.root_flow, scene).glob("*.flo"))

    def image_files(self, index: int) -> Tuple:
        return self._image_list[index]

    def flow_file(self, index: int) -> Optional[Union[str, Path]]:
        if self.split == "test":
            return None
        return self._flow_list[index]

    def read_flow(self, filename: Union[str, Path]) -> Tuple[Tensor, None]:
        flow = optical_flow.read(filename, fmt="middlebury")
        valid = None
        return flow, valid

    def __len__(self):
        return len(self._image_list)

from pathlib import Path
from typing import Optional, Tuple, Union

from torch import Tensor

import optical_flow
from optical_flow.datasets.base import OpticalFlowDataset


class KITTI(OpticalFlowDataset):
    def __init__(self, root: Union[str, Path], split: str = "training"):
        super().__init__()
        self.root = root
        self.split = split
        self._flow_list = sorted(Path(root, split, "flow_occ").glob("*_10.png"))
        self._image_list = []
        self._extra_info = []

        images1 = sorted(Path(root, split, "image_2").glob("*_10.png"))
        images2 = sorted(Path(root, split, "image_2").glob("*_11.png"))

        for img1, img2 in zip(images1, images2):
            frame_id = img1.with_suffix("").name[-1]
            self._extra_info += [[frame_id]]
            self._image_list += [(img1, img2)]

    def image_files(self, index: int) -> Tuple:
        return self._image_list[index]

    def flow_file(self, index: int) -> Optional[Union[str, Path]]:
        return self._flow_list[index] if self.split == "training" else None

    def read_flow(self, filename: Union[str, Path]) -> Tuple[Tensor, Tensor]:
        flow, valid = optical_flow.read(filename, fmt="kitti", return_mask=True)
        return flow, valid

    def __len__(self):
        return len(self._image_list)

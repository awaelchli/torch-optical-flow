from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from torch import Tensor

import optical_flow
from optical_flow.datasets.base import OpticalFlowDataset

TRAIN_VAL_SPLIT_FILE = Path(__file__).parent / "split_files" / "chairs_split.txt"


class FlyingChairs(OpticalFlowDataset):
    def __init__(self, root: Union[str, Path], split="training"):
        super().__init__()
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


class FlyingThings3D(OpticalFlowDataset):
    def __init__(self, root: Union[str, Path], dstype: str = "frames_cleanpass"):
        super().__init__()
        self.root = root
        self.dstype = dstype
        self._image_list = []
        self._flow_list = []

        for cam in ["left"]:
            for direction in ["into_future", "into_past"]:
                # sorted(Path(root, dstype, "TRAIN").rglob("*/*"))
                image_dirs = sorted(glob(osp.join(root, dstype, "TRAIN/*/*")))
                image_dirs = sorted([osp.join(f, cam) for f in image_dirs])

                flow_dirs = sorted(glob(osp.join(root, "optical_flow/TRAIN/*/*")))
                flow_dirs = sorted([osp.join(f, direction, cam) for f in flow_dirs])

                for idir, fdir in zip(image_dirs, flow_dirs):
                    images = sorted(glob(osp.join(idir, "*.png")))
                    flows = sorted(glob(osp.join(fdir, "*.pfm")))
                    for i in range(len(flows) - 1):
                        if direction == "into_future":
                            self._image_list += [(images[i], images[i + 1])]
                            self._flow_list += [flows[i]]
                        elif direction == "into_past":
                            self._image_list += [(images[i + 1], images[i])]
                            self._flow_list += [flows[i + 1]]

    def image_files(self, index: int) -> Tuple:
        return self._image_list[index]

    def flow_file(self, index: int) -> Optional[Union[str, Path]]:
        return self._flow_list[index]

    def read_flow(self, filename: Union[str, Path]) -> Tuple[Tensor, None]:
        flow = optical_flow.read(filename, fmt="pfm")
        valid = None
        return flow, valid

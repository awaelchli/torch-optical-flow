from pathlib import Path
from typing import Optional, Tuple, Union

from torch import Tensor

import optical_flow
from optical_flow.datasets.base import OpticalFlowDataset


class HD1K(OpticalFlowDataset):
    def __init__(self, root: Union[str, Path]):
        super().__init__()

        self._image_list = []
        self._flow_list = []

        seq_ix = 0
        while 1:
            flows = sorted(
                Path(root, "hd1k_flow_gt", "flow_occ").glob(f"{seq_ix:06d}_*.png")
            )
            images = sorted(
                Path(root, "hd1k_input", "image_2").glob(f"{seq_ix:06d}_*.png")
            )

            if len(flows) == 0:
                break

            for i in range(len(flows) - 1):
                self._flow_list += [flows[i]]
                self._image_list += [(images[i], images[i + 1])]

            seq_ix += 1

    def image_files(self, index: int) -> Tuple:
        return self._image_list[index]

    def flow_file(self, index: int) -> Optional[Union[str, Path]]:
        return self._flow_list[index]

    def read_flow(self, filename: Union[str, Path]) -> Tuple[Tensor, Tensor]:
        flow, valid = optical_flow.read(filename, fmt="kitti", return_mask=True)
        return flow, valid

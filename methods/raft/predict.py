import os
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torchvision
from jsonargparse import CLI
from model import RAFT
from model.utils import InputPadder
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

import optical_flow
from tqdm import tqdm


class FlowInferenceDataset(Dataset):
    def __init__(self, folder: Union[Path, str], ext: Optional[str] = None) -> None:
        self.files = (
            sorted(Path(folder, p) for p in os.listdir(folder))
            if ext is None
            else Path(folder).glob(f"*.{ext}")
        )

    def __getitem__(self, item: int) -> Tuple[Tensor, Tensor]:
        img0 = Image.open(self.files[item])
        img1 = Image.open(self.files[item + 1])
        img0 = torch.tensor(np.array(img0)).permute(2, 0, 1)[:3].float()
        img1 = torch.tensor(np.array(img1)).permute(2, 0, 1)[:3].float()
        return img0, img1

    def __len__(self) -> int:
        return len(self.files) - 1


@torch.inference_mode()
def main(
    source: str,
    destination: str,
    checkpoint: str = "pretrained/checkpoints/raft-sintel.ckpt",
    ext: Optional[str] = None,
    overwrite: bool = False,
    iters: int = 24,
    visualize: bool = True,
):
    """Predict optical flow with the RAFT model.

    This command takes a path to a folder with images as input and will predict the optical flow between
    consecutive pairs of images (in the order sorted by filename). The raw optical flow is written to a
    .flo file in Middlebury format.

    Args:
        source: The source folder with 8-bit RGB images.
        destination: The destination path to write optical flow predictions. If the destination does
            not exist, it will be created. If the destination path exists and is not empty, an error
            will be produce unless the argument ``overwrite`` is set.
        checkpoint: The checkpoint file to load model weights from. By default, will load weights
            trained on the FlyingChairs, FlyingThings3D and Sintel datasets.
        ext: File extension to filter input images from the source directory. If not provided
            every file in the source directory will be considered an RGB image.
        overwrite: When set to ``True``, overwrite files at the destination.
        iters: The number of recurrent steps for flow refinement. Must be a positive integer.
        visualize: For every output, save a PNG file containing a image grid with: first image, 
            second image, RGB optical flow.
    """
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=overwrite)

    dataset = FlowInferenceDataset(source, ext=ext)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=4)
    device = (
        torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    )

    model = RAFT.load_from_checkpoint(checkpoint)
    model.to(device)

    for i, (img0, img1) in tqdm(enumerate(dataloader), total=len(dataset)):
        img0, img1 = img0.to(device), img1.to(device)

        padder = InputPadder(img0.shape)
        padded0, padded1 = padder.pad(img0, img1)
        _, flow = model(padded0, padded1, iters=iters, test_mode=True)

        assert flow.shape[0] == 1
        flow = padder.unpad(flow)[0]
        
        flow_raw_file = destination / f"{i:06d}.flo"
        optical_flow.write(flow_raw_file, flow)

        if visualize:
            img0 = img0[0] / 255.0
            img1 = img1[0] / 255.0
            flow_rgb = optical_flow.flow2rgb(flow)
            flow_rgb_file = flow_raw_file.with_suffix(".png")
            torchvision.utils.save_image([img0, img1, flow_rgb], flow_rgb_file)


if __name__ == "__main__":
    CLI(main)

import optical_flow
import os
from pathlib import Path
from typing import Sequence, Any, Optional, Tuple, Union

import torch
from PIL import Image
from pytorch_lightning import Trainer, LightningModule
from torch import Tensor
from torch.utils.data import Dataset, IterableDataset
from torch.utils.data.dataloader import DataLoader

from cli import RAFTCLI
from data.datamodule import RAFTDataModule
from model import RAFT
import numpy as np
from model.utils import InputPadder
import torchvision
from pytorch_lightning.callbacks import BasePredictionWriter


class FlowInferenceDataset(Dataset):

    def __init__(self, folder: Union[Path, str], ext: Optional[str] = None):
        self.files = sorted(Path(folder, p) for p in os.listdir(folder)) if ext is None else Path(folder).glob(f"*.{ext}")

    def __getitem__(self, item: int) -> Tuple[Tensor, Tensor]:
        img0 = Image.open(self.files[item])
        img1 = Image.open(self.files[item + 1])
        img0 = torch.tensor(np.array(img0)).permute(2, 0, 1)[:3].float()
        img1 = torch.tensor(np.array(img1)).permute(2, 0, 1)[:3].float()
        return img0, img1

    def __len__(self):
        return len(self.files) - 1


class RAFTPredictionCLI(RAFTCLI):
    def fit(self) -> None:
        self.fit_kwargs.update(return_predictions=False)
        self.trainer.predict(**self.fit_kwargs)


# def main():
#     cli = RAFTPredictionCLI(
#         RAFT,
#         RAFTDataModule,
#         description="Lightning RAFT Prediction",
#         parser_kwargs={"default_config_files": ["config/predict/default.yaml"]},
#         save_config_callback=None,
#     )


@torch.inference_mode()
def main(folder: str, output: Optional[str] = None, ext: Optional[str] = None, overwrite: bool = False, checkpoint: str = "pretrained/checkpoints/raft-sintel.ckpt"):
    output_folder = Path.cwd() / "predictions" if output is None else Path(output)
    output_folder.mkdir(parents=True, exist_ok=overwrite)

    dataset = FlowInferenceDataset(folder, ext=ext)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=8)
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    
    model = RAFT.load_from_checkpoint(checkpoint)
    model.to(device)
    
    for i, (img0, img1) in enumerate(dataloader):
        img0, img1 = img0.to(device), img1.to(device)
        
        padder = InputPadder(img0.shape)
        padded0, padded1 = padder.pad(img0, img1)
        _, flow_pr = model(padded0, padded1, iters=32, test_mode=True)
        
        assert flow_pr.shape[0] == 1
        flow_pr = padder.unpad(flow_pr)[0]
        flow_rgb = optical_flow.flow2rgb(flow_pr)


        img0 = img0[0] / 255.0
        img1 = img1[0] / 255.0


        flow_out = output_folder / f"{i:05d}.flo"
        flow_rgb_out = flow_out.with_suffix(".png")
        optical_flow.write(flow_out, flow_pr)
        torchvision.utils.save_image([img0, img1, flow_rgb], flow_rgb_out)
        
        


        


#
# def create_kitti_submission(model, iters=24, output_path='kitti_submission'):
#     """ Create submission for the Sintel leaderboard """
#     model.eval()
#     test_dataset = datasets.KITTI(split='testing', aug_params=None)
#
#     if not os.path.exists(output_path):
#         os.makedirs(output_path)
#
#     for test_id in range(len(test_dataset)):
#         image1, image2, (frame_id,) = test_dataset[test_id]
#         padder = InputPadder(image1.shape, mode='kitti')
#         image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())
#
#         _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
#         flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()
#
#         output_filename = os.path.join(output_path, frame_id)
#         frame_utils.writeFlowKITTI(output_filename, flow)

#
# def create_sintel_submission(model, iters=32, warm_start=False, output_path='sintel_submission'):
#     """ Create submission for the Sintel leaderboard """
#     model.eval()
#     for dstype in ['clean', 'final']:
#         test_dataset = datasets.MpiSintel(split='test', aug_params=None, dstype=dstype)
#
#         flow_prev, sequence_prev = None, None
#         for test_id in range(len(test_dataset)):
#             image1, image2, (sequence, frame) = test_dataset[test_id]
#             if sequence != sequence_prev:
#                 flow_prev = None
#
#             padder = InputPadder(image1.shape)
#             image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())
#
#             flow_low, flow_pr = model(image1, image2, iters=iters, flow_init=flow_prev, test_mode=True)
#             flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()
#
#             if warm_start:
#                 flow_prev = forward_interpolate(flow_low[0])[None].cuda()
#
#             output_dir = os.path.join(output_path, dstype, sequence)
#             output_file = os.path.join(output_dir, 'frame%04d.flo' % (frame + 1))
#
#             if not os.path.exists(output_dir):
#                 os.makedirs(output_dir)
#
#             frame_utils.writeFlow(output_file, flow)
#             sequence_prev = sequence


if __name__ == "__main__":
    main(folder="/Volumes/Archive/Datasets/MPI-Sintel/training/final/market_2", overwrite=True)
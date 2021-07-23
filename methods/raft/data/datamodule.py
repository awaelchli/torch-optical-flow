import torch
from data.dataset import FlyingChairs, FlyingThings3D, HD1K, KITTI, MpiSintel
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader


class RAFTDataModule(LightningDataModule):
    def __init__(
        self,
        stage: str = "chairs",
        image_size: tuple = (384, 512),
        batch_size: int = 6,
        num_workers: int = 4,
        root_chairs: str = "datasets/FlyingChairs_release/data",
        root_things: str = "datasets/FlyingThings3D",
        root_sintel: str = "datasets/Sintel",
        root_kitti: str = "datasets/KITTI",
        root_hd1k: str = "datasets/HD1k",
    ):
        super().__init__()
        self.stage = stage
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        print(root_chairs)
        self.root_chairs = root_chairs
        self.root_things = root_things
        self.root_sintel = root_sintel
        self.root_kitti = root_kitti
        self.root_hd1k = root_hd1k

    def train_dataloader(self):
        train_ds = "C+T+K+S+H"

        if self.stage == "chairs":
            aug_params = {
                "crop_size": self.image_size,
                "min_scale": -0.1,
                "max_scale": 1.0,
                "do_flip": True,
            }
            train_dataset = FlyingChairs(
                aug_params, split="training", root=self.root_chairs
            )

        elif self.stage == "things":
            aug_params = {
                "crop_size": self.image_size,
                "min_scale": -0.4,
                "max_scale": 0.8,
                "do_flip": True,
            }
            clean_dataset = FlyingThings3D(
                aug_params, dstype="frames_cleanpass", root=self.root_things
            )
            final_dataset = FlyingThings3D(
                aug_params, dstype="frames_finalpass", root=self.root_things
            )
            train_dataset = clean_dataset + final_dataset

        elif self.stage == "sintel":
            aug_params = {
                "crop_size": self.image_size,
                "min_scale": -0.2,
                "max_scale": 0.6,
                "do_flip": True,
            }
            things = FlyingThings3D(
                aug_params, dstype="frames_cleanpass", root=self.root_things
            )
            sintel_clean = MpiSintel(
                aug_params, split="training", dstype="clean", root=self.root_sintel
            )
            sintel_final = MpiSintel(
                aug_params, split="training", dstype="final", root=self.root_sintel
            )

            if train_ds == "C+T+K+S+H":
                kitti = KITTI(
                    {
                        "crop_size": self.image_size,
                        "min_scale": -0.3,
                        "max_scale": 0.5,
                        "do_flip": True,
                    },
                    root=self.root_kitti,
                )
                hd1k = HD1K(
                    {
                        "crop_size": self.image_size,
                        "min_scale": -0.5,
                        "max_scale": 0.2,
                        "do_flip": True,
                    },
                    root=self.root_hd1k,
                )
                train_dataset = (
                    100 * sintel_clean
                    + 100 * sintel_final
                    + 200 * kitti
                    + 5 * hd1k
                    + things
                )

            elif train_ds == "C+T+K/S":
                train_dataset = 100 * sintel_clean + 100 * sintel_final + things

        elif self.stage == "kitti":
            aug_params = {
                "crop_size": self.image_size,
                "min_scale": -0.2,
                "max_scale": 0.4,
                "do_flip": False,
            }
            train_dataset = KITTI(aug_params, split="training", root=self.root_kitti)

        dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=torch.cuda.is_available(),
            num_workers=self.num_workers,
        )

        return dataloader

    def val_dataloader(self):
        # TODO: validation for other stages
        val_dataset = FlyingChairs(split="validation", root=self.root_chairs)
        dataloader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
            num_workers=self.num_workers,
        )
        return dataloader

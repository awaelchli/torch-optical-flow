from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from data.dataset import FlyingChairs, FlyingThings3D, MpiSintel, KITTI, HD1K


class RAFTDataModule(LightningDataModule):
    def __init__(
        self,
        stage: str = "chairs",
        image_size: tuple = (384, 512),
        batch_size: int = 6,
    ):
        super().__init__()
        self.stage = stage
        self.image_size = image_size
        self.batch_size = batch_size

    def train_dataloader(self):
        TRAIN_DS = "C+T+K+S+H"

        if self.stage == "chairs":
            aug_params = {
                "crop_size": self.image_size,
                "min_scale": -0.1,
                "max_scale": 1.0,
                "do_flip": True,
            }
            train_dataset = FlyingChairs(aug_params, split="training")

        elif self.stage == "things":
            aug_params = {
                "crop_size": self.image_size,
                "min_scale": -0.4,
                "max_scale": 0.8,
                "do_flip": True,
            }
            clean_dataset = FlyingThings3D(aug_params, dstype="frames_cleanpass")
            final_dataset = FlyingThings3D(aug_params, dstype="frames_finalpass")
            train_dataset = clean_dataset + final_dataset

        elif self.stage == "sintel":
            aug_params = {
                "crop_size": self.image_size,
                "min_scale": -0.2,
                "max_scale": 0.6,
                "do_flip": True,
            }
            things = FlyingThings3D(aug_params, dstype="frames_cleanpass")
            sintel_clean = MpiSintel(aug_params, split="training", dstype="clean")
            sintel_final = MpiSintel(aug_params, split="training", dstype="final")

            if TRAIN_DS == "C+T+K+S+H":
                kitti = KITTI(
                    {
                        "crop_size": self.image_size,
                        "min_scale": -0.3,
                        "max_scale": 0.5,
                        "do_flip": True,
                    }
                )
                hd1k = HD1K(
                    {
                        "crop_size": self.image_size,
                        "min_scale": -0.5,
                        "max_scale": 0.2,
                        "do_flip": True,
                    }
                )
                train_dataset = (
                    100 * sintel_clean
                    + 100 * sintel_final
                    + 200 * kitti
                    + 5 * hd1k
                    + things
                )

            elif TRAIN_DS == "C+T+K/S":
                train_dataset = 100 * sintel_clean + 100 * sintel_final + things

        elif self.stage == "kitti":
            aug_params = {
                "crop_size": self.image_size,
                "min_scale": -0.2,
                "max_scale": 0.4,
                "do_flip": False,
            }
            train_dataset = KITTI(aug_params, split="training")

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            pin_memory=False,
            shuffle=True,
            num_workers=4,
            drop_last=True,
        )

        return train_loader

import torch
from data.datamodule import RAFTDataModule
from jsonargparse import CLI
from model import RAFT
from pretrained.convert import strip_module
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger


def main(checkpoint: str):
    model = RAFT()
    datamodule = RAFTDataModule()

    # TODO: use load_from_checkpoint
    checkpoint = torch.load(checkpoint)
    state_dict = checkpoint.get("state_dict", checkpoint)
    state_dict = strip_module(state_dict)
    model.load_state_dict(state_dict, strict=False)

    trainer = Trainer(
        gpus=1,
        logger=WandbLogger(
            save_dir="./logs",
            project="lightning-raft",
            name=f"raft-evaluation",
        ),
    )
    trainer.validate(model, datamodule)


if __name__ == "__main__":
    CLI()

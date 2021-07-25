import torch
from data.datamodule import RAFTDataModule
from model import RAFT
from pytorch_lightning import seed_everything
from pytorch_lightning.utilities.cli import LightningArgumentParser, LightningCLI
from pytorch_lightning.loggers import WandbLogger


class RAFTCLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.add_argument("--restore_ckpt", help="restore checkpoint")

    def before_fit(self) -> None:
        if self.config["restore_ckpt"] is not None:
            self.model.load_state_dict(
                torch.load(self.config["restore_ckpt"]), strict=False
            )

        if self.datamodule.stage != "chairs":
            self.model.freeze_bn()


def main():
    seed_everything(1234)
    cli = RAFTCLI(
        RAFT,
        RAFTDataModule,
        description="Lightning RAFT",
        save_config_callback=None,
    )

if __name__ == "__main__":
    main()

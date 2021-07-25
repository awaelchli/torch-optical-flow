import torch
from data.datamodule import RAFTDataModule
from model import RAFT
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.cli import LightningArgumentParser, LightningCLI
from pytorch_lightning.utilities import rank_zero_info
from pretrained.convert import strip_module


class RAFTCLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.add_argument("--restore_ckpt", help="restore checkpoint")

    def before_fit(self) -> None:
        restore_path = self.config["restore_ckpt"]
        if restore_path is not None:
            state_dict = torch.load(restore_path)
            state_dict = strip_module(state_dict)
            self.model.load_state_dict(state_dict, strict=False)
            rank_zero_info(f"Restored weights from {restore_path}")

        if self.datamodule.stage != "chairs":
            self.model.freeze_bn()


def main():
    cli = RAFTCLI(
        RAFT,
        RAFTDataModule,
        description="Lightning RAFT",
        save_config_callback=None,
    )


if __name__ == "__main__":
    main()

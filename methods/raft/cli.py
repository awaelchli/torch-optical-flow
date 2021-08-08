import torch
from pretrained.convert import strip_module
from pytorch_lightning.utilities import rank_zero_info
from pytorch_lightning.utilities.cli import LightningArgumentParser, LightningCLI


class RAFTCLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.add_argument("--restore_weights", help="restore checkpoint")

    def before_fit(self) -> None:
        restore_path = self.config["restore_weights"]
        if restore_path is not None:
            checkpoint = torch.load(restore_path)
            state_dict = checkpoint.get("state_dict", checkpoint)
            state_dict = strip_module(state_dict)
            self.model.load_state_dict(state_dict, strict=False)
            rank_zero_info(f"Restored weights from {restore_path}")

        if self.datamodule.stage != "chairs":
            self.model.freeze_bn()

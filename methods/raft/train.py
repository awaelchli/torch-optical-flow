from cli import RAFTCLI
from data.datamodule import RAFTDataModule
from model import RAFT


def main():
    cli = RAFTCLI(
        RAFT,
        RAFTDataModule,
        description="Lightning RAFT Training",
        parser_kwargs={"default_config_files": ["config/train/default.yaml"]},
        save_config_callback=None,
    )


if __name__ == "__main__":
    main()

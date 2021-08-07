from data.datamodule import RAFTDataModule
from model import RAFT
from cli import RAFTCLI


def main():
    cli = RAFTCLI(
        RAFT,
        RAFTDataModule,
        description="Lightning RAFT Training",
        save_config_callback=None,
    )


if __name__ == "__main__":
    main()

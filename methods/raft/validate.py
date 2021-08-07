from data.datamodule import RAFTDataModule
from model import RAFT
from cli import RAFTCLI


class RAFTEvaluationCLI(RAFTCLI):

    def fit(self) -> None:
        self.trainer.validate(**self.fit_kwargs)


def main():
    cli = RAFTEvaluationCLI(
        RAFT,
        RAFTDataModule,
        description="Lightning RAFT Evaluation",
        save_config_callback=None,
    )


if __name__ == "__main__":
    main()

from cli import RAFTCLI
from data.datamodule import RAFTDataModule
from model import RAFT


class RAFTEvaluationCLI(RAFTCLI):
    def fit(self) -> None:
        self.trainer.validate(**self.fit_kwargs)


def main():
    cli = RAFTEvaluationCLI(
        RAFT,
        RAFTDataModule,
        description="Lightning RAFT Evaluation",
        parser_kwargs={"default_config_files": ["config/default.yaml"]},
        save_config_callback=None,
    )


if __name__ == "__main__":
    main()

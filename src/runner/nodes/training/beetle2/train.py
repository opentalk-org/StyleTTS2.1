import argparse
from pathlib import Path

from .logger import configure_logging, logger
from .trainer import Trainer


class ConsoleCallbacks:
    def check_cancel(self) -> None:
        return

    def report_index_progress(self, scanned: int, total: int) -> None:
        logger.info("index %d/%d", scanned, total)

    def report_training_progress(self, step: int, total: int) -> None:
        logger.info("training %d/%d", step, total)


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(prog="beetle2-train")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--initialize", type=Path)
    arguments = parser.parse_args()
    trainer = Trainer(
        config_path=arguments.config,
        output_path=arguments.output,
        callbacks=ConsoleCallbacks(),
        resume_path=arguments.resume,
        initialize_path=arguments.initialize,
    )
    completed = trainer.train()
    logger.info("completed %d optimizer steps", completed)


if __name__ == "__main__":
    main()

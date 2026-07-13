import argparse
import asyncio
from uuid import uuid4

from runner.worker import RunnerWorker
from shared.logging_setup import configure_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner-id", default=f"runner_{uuid4().hex[:8]}")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging("runner")
    worker = RunnerWorker(runner_id=args.runner_id)
    asyncio.run(worker.run())


if __name__ == "__main__":
    main()

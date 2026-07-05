import argparse
import asyncio
from uuid import uuid4

from runner.worker import RunnerWorker
from shared.jetstream import DEFAULT_NATS_URL
from shared.logging_setup import configure_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner-id", default=f"runner_{uuid4().hex[:8]}")
    parser.add_argument("--nats-url", default=DEFAULT_NATS_URL)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging("runner")
    worker = RunnerWorker(runner_id=args.runner_id, nats_url=args.nats_url)
    asyncio.run(worker.run())


if __name__ == "__main__":
    main()

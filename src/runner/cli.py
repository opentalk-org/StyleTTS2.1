import argparse
import asyncio
import os
from uuid import uuid4

from shared.logging_setup import configure_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner-id", default=f"runner_{uuid4().hex[:8]}")
    parser.add_argument(
        "--gpu-index",
        type=int,
        default=None,
        help="Physical GPU index to expose to this runner.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.gpu_index is not None:
        if args.gpu_index < 0:
            raise ValueError("--gpu-index must be non-negative")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)
        os.environ["RUNFLOW_RUNNER_GPU_INDEX"] = str(args.gpu_index)
    from runner.worker import RunnerWorker

    configure_logging("runner")
    worker = RunnerWorker(runner_id=args.runner_id)
    asyncio.run(worker.run())


if __name__ == "__main__":
    main()

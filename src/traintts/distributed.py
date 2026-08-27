import argparse
import logging

import torch.distributed as dist

from .main import LOG_FORMAT
from .train import train


def main() -> None:
    parser = argparse.ArgumentParser(prog="finetune-distributed")
    parser.add_argument("config")
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    train(arguments.config, run=None)
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

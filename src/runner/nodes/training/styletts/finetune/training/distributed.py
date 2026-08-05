import argparse

import torch.distributed as dist

from .train_finetune import train


def main() -> None:
    parser = argparse.ArgumentParser(prog="finetune-distributed")
    parser.add_argument("config")
    arguments = parser.parse_args()
    train(arguments.config, run=None)
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

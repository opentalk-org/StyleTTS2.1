import argparse
import logging

from givemedata_client import GiveMeDataClient
import torch.distributed as dist

from .main import LOG_FORMAT
from .train import train


def main() -> None:
    parser = argparse.ArgumentParser(prog="finetune-distributed")
    parser.add_argument("config")
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    data_client = GiveMeDataClient()
    try:
        train(arguments.config, run=None, data_client=data_client)
    finally:
        try:
            data_client.close()
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()


if __name__ == "__main__":
    main()

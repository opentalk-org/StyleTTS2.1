import os

from givemedata_client import GiveMeDataClient

from traintts.train import train as train_with_client


def train(config_path: str) -> None:
    data_client = GiveMeDataClient(os.environ["GIVEMEDATA_TRAINING_ID"])
    try:
        train_with_client(config_path, run=None, data_client=data_client)
    finally:
        data_client.close()

import os
import queue
import shutil
import tarfile
from collections.abc import Iterator
from pathlib import Path

import grpc

from . import givemedata_pb2 as pb
from . import givemedata_pb2_grpc as pb_grpc

DEFAULT_ADDR = "localhost:8181"


class GiveMeDataClient:
    def __init__(self, addr: str | None = None) -> None:
        addr = addr or os.environ.get("GIVEMEDATA_ADDR", DEFAULT_ADDR)
        self._channel = grpc.insecure_channel(addr, options=[("grpc.max_receive_message_length", 67136000)])
        self._stub = pb_grpc.GiveMeDataStub(self._channel)
        response = self._stub.Init(pb.InitRequest())
        self.session_id: str = response.session_id
        # verbatim yaml the service passes through, untouched, for the training loop
        self.train_config: str = response.train_config

    def batches(self, split: int, prefetch: int = 4) -> Iterator[pb.DataResponse]:
        requests: queue.SimpleQueue[pb.DataRequest] = queue.SimpleQueue()
        request = pb.DataRequest(session_id=self.session_id, split=split)
        for _ in range(prefetch):
            requests.put(request)

        def request_iterator() -> Iterator[pb.DataRequest]:
            while True:
                yield requests.get()

        for response in self._stub.Data(request_iterator()):
            requests.put(request)
            yield response

    def download_asset(self, name: str, dest_dir: Path) -> Path:
        """Fetch one named asset into <dest_dir>/<name>/, skipping the download
        when a previous run already completed it (the .done marker)."""
        asset_dir = dest_dir / name
        marker = dest_dir / f"{name}.done"
        if marker.exists():
            return asset_dir

        dest_dir.mkdir(parents=True, exist_ok=True)
        part = dest_dir / f"{name}.part"
        with open(part, "wb") as f:
            for response in self._stub.Asset(
                pb.AssetRequest(session_id=self.session_id, name=name)
            ):
                f.write(response.chunk)

        shutil.rmtree(asset_dir, ignore_errors=True)
        asset_dir.mkdir(parents=True)
        if tarfile.is_tarfile(part):
            with tarfile.open(part) as tar:
                tar.extractall(asset_dir, filter="data")
            part.unlink()
        else:
            part.rename(asset_dir / name)
        marker.touch()
        return asset_dir

    def close(self) -> None:
        self._stub.End(pb.EndRequest(session_id=self.session_id))
        self._channel.close()


def asset_file(asset_dir: Path) -> Path:
    """The single file inside an extracted asset; errors if there isn't exactly one."""
    files = [p for p in asset_dir.rglob("*") if p.is_file()]
    if len(files) != 1:
        raise ValueError(f"expected exactly one file in {asset_dir}, found {len(files)}")
    return files[0]

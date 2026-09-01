import os
import queue
import shutil
import tarfile
import tempfile
import threading
from collections.abc import Iterator
from pathlib import Path

import grpc

from . import givemedata_pb2 as pb
from . import givemedata_pb2_grpc as pb_grpc
from .metrics import MetricsStream

DEFAULT_ADDR = "localhost:8181"
CHECKPOINT_CHUNK_BYTES = 2 * 1024 * 1024


class GiveMeDataClient:
    def __init__(self, training_id: str, addr: str | None = None) -> None:
        addr = addr or os.environ.get("GIVEMEDATA_ADDR", DEFAULT_ADDR)
        self._channel = grpc.insecure_channel(addr, options=[("grpc.max_receive_message_length", 67136000)])
        self._stub = pb_grpc.GiveMeDataStub(self._channel)
        response = self._stub.Init(pb.InitRequest(training_id=training_id))
        self.training_id: str = response.training_id
        # verbatim yaml the service passes through, untouched, for the training loop
        self.train_config: str = response.train_config
        self._metrics: MetricsStream | None = None
        self._closed = False

    def batches(self, split: int, prefetch: int = 4) -> Iterator[pb.DataResponse]:
        requests: queue.SimpleQueue[pb.DataRequest | None] = queue.SimpleQueue()
        stopped = threading.Event()
        request = pb.DataRequest(training_id=self.training_id, split=split)
        for _ in range(prefetch):
            requests.put(request)

        def request_iterator() -> Iterator[pb.DataRequest]:
            while not stopped.is_set():
                queued = requests.get()
                if queued is None:
                    return
                yield queued

        try:
            for response in self._stub.Data(request_iterator()):
                requests.put(request)
                yield response
        finally:
            stopped.set()
            requests.put(None)

    def download_asset(self, name: str, dest_dir: Path) -> Path:
        """Fetch one named asset and return its configured entrypoint or directory."""
        asset_dir = dest_dir / name
        marker = dest_dir / f"{name}.done"
        if marker.exists():
            return _resolved_asset_path(asset_dir, marker.read_text().strip())

        dest_dir.mkdir(parents=True, exist_ok=True)
        part = dest_dir / f"{name}.part"
        responses = iter(
            self._stub.Asset(pb.AssetRequest(training_id=self.training_id, name=name))
        )
        first = next(responses)
        if first.WhichOneof("payload") != "metadata":
            raise ValueError(f"asset {name!r} stream did not start with metadata")
        metadata = first.metadata
        entrypoint = (
            metadata.entrypoint if metadata.HasField("entrypoint") else None
        )
        with open(part, "wb") as f:
            for response in responses:
                if response.WhichOneof("payload") != "chunk":
                    raise ValueError(f"asset {name!r} stream contained repeated metadata")
                f.write(response.chunk)

        shutil.rmtree(asset_dir, ignore_errors=True)
        asset_dir.mkdir(parents=True)
        if tarfile.is_tarfile(part):
            with tarfile.open(part) as tar:
                tar.extractall(asset_dir, filter="data")
            part.unlink()
        else:
            part.rename(asset_dir / name)
        relative_path = entrypoint or "."
        resolved = _resolved_asset_path(asset_dir, relative_path)
        if not resolved.exists():
            raise FileNotFoundError(
                f"asset {name!r} entrypoint {relative_path!r} does not exist"
            )
        marker.write_text(relative_path)
        return resolved

    def upload_checkpoint(self, step: int, source_dir: Path) -> None:
        """Stream a checkpoint directory to the service as a tar (the same
        shape assets have, so it can come back as a base checkpoint later)."""
        # tar to a file, not memory: checkpoints are GBs
        with tempfile.NamedTemporaryFile(
            dir=source_dir.parent, suffix=".tar", delete=False
        ) as tmp:
            tar_path = Path(tmp.name)
        try:
            with tarfile.open(tar_path, "w") as tar:
                for entry in sorted(source_dir.iterdir()):
                    tar.add(entry, arcname=entry.name)

            def requests() -> Iterator[pb.CheckpointRequest]:
                yield pb.CheckpointRequest(
                    metadata=pb.CheckpointMetadata(training_id=self.training_id, step=step)
                )
                with open(tar_path, "rb") as f:
                    while chunk := f.read(CHECKPOINT_CHUNK_BYTES):
                        yield pb.CheckpointRequest(chunk=chunk)

            self._stub.Checkpoint(requests())
        finally:
            tar_path.unlink(missing_ok=True)

    def metrics(self) -> MetricsStream:
        if self._closed:
            raise RuntimeError("givemedata client is closed")
        if self._metrics is None:
            self._metrics = MetricsStream(self._stub, self.training_id)
        return self._metrics

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        metrics_error: BaseException | None = None
        try:
            if self._metrics is not None:
                self._metrics.close()
        except BaseException as error:
            metrics_error = error
        try:
            self._stub.End(pb.EndRequest(training_id=self.training_id))
        finally:
            self._channel.close()
        if metrics_error is not None:
            raise metrics_error


def _resolved_asset_path(asset_dir: Path, relative_path: str) -> Path:
    if not relative_path:
        raise ValueError(f"asset marker {asset_dir}.done is empty")
    return asset_dir if relative_path == "." else asset_dir / relative_path

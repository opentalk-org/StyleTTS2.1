import os
import queue
from collections.abc import Iterator

import grpc

from . import givemedata_pb2 as pb
from . import givemedata_pb2_grpc as pb_grpc

DEFAULT_ADDR = "localhost:8181"


class GiveMeDataClient:
    def __init__(
        self,
        dataset_id: str,
        *,
        validation_samples: int,
        max_seconds: float,
        max_text_tokens: int,
        plbert_languages: list[str] | None = None,
        plbert_modality_id: int = 0,
        seed: int = 1,
        addr: str | None = None,
    ) -> None:
        self.plbert_modality_id = plbert_modality_id
        addr = addr or os.environ.get("GIVEMEDATA_ADDR", DEFAULT_ADDR)
        self._channel = grpc.insecure_channel(addr, options=[("grpc.max_receive_message_length", 67136000)])
        self._stub = pb_grpc.GiveMeDataStub(self._channel)
        response = self._stub.Init(
            pb.InitRequest(
                dataset_id=dataset_id,
                validation_samples=validation_samples,
                seed=seed,
                max_seconds=max_seconds,
                max_text_tokens=max_text_tokens,
                plbert_languages=plbert_languages or [],
                plbert_modality_id=plbert_modality_id,
            )
        )
        self.session_id: str = response.session_id

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

    def close(self) -> None:
        self._stub.End(pb.EndRequest(session_id=self.session_id))
        self._channel.close()

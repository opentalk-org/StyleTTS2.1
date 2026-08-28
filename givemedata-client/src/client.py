import os
import queue
from collections.abc import Iterator

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

    def close(self) -> None:
        self._stub.End(pb.EndRequest(session_id=self.session_id))
        self._channel.close()

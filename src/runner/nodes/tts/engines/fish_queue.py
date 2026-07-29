from __future__ import annotations

from dataclasses import dataclass
import logging
import queue
import threading
import traceback
from typing import Any, Callable

import torch

FISH_MAX_SEQUENCE_LENGTH = 4096
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class FishQueueDependencies:
    init_model: Callable[..., tuple[Any, Callable[..., Any]]]
    generate_long: Callable[..., Any]
    wrapped_response: Callable[..., Any]


def launch_memory_bounded_queue(
    checkpoint_path: str,
    device: str,
    precision: torch.dtype,
    dependencies: FishQueueDependencies,
) -> queue.Queue[Any]:
    input_queue: queue.Queue[Any] = queue.Queue()
    initialized = threading.Event()
    failures: queue.Queue[BaseException] = queue.Queue()
    worker = threading.Thread(
        target=_queue_worker,
        args=(
            checkpoint_path,
            device,
            precision,
            dependencies,
            input_queue,
            initialized,
            failures,
        ),
        daemon=True,
    )
    worker.start()
    initialized.wait()
    if not failures.empty():
        raise failures.get()
    return input_queue


def _queue_worker(
    checkpoint_path: str,
    device: str,
    precision: torch.dtype,
    dependencies: FishQueueDependencies,
    input_queue: queue.Queue[Any],
    initialized: threading.Event,
    failures: queue.Queue[BaseException],
) -> None:
    try:
        model, decode_one_token = dependencies.init_model(
            checkpoint_path, device, precision, compile=False
        )
        assert model.config.max_seq_len >= FISH_MAX_SEQUENCE_LENGTH, (
            "Fish Speech checkpoint sequence limit is below the corpus runtime limit"
        )
        model.config.max_seq_len = FISH_MAX_SEQUENCE_LENGTH
        with torch.device(device):
            model.setup_caches(
                max_batch_size=1,
                max_seq_len=FISH_MAX_SEQUENCE_LENGTH,
                dtype=next(model.parameters()).dtype,
            )
    except BaseException as exc:
        failures.put(exc)
        initialized.set()
        return

    initialized.set()
    while True:
        item = input_queue.get()
        if item is None:
            return
        try:
            for chunk in dependencies.generate_long(
                model=model,
                decode_one_token=decode_one_token,
                **item.request,
            ):
                item.response_queue.put(
                    dependencies.wrapped_response(
                        status="success",
                        response=chunk,
                    )
                )
            torch.cuda.empty_cache()
        except Exception as exc:
            LOGGER.error("Fish Speech generation failed\n%s", traceback.format_exc())
            item.response_queue.put(
                dependencies.wrapped_response(status="error", response=exc)
            )
            torch.cuda.empty_cache()

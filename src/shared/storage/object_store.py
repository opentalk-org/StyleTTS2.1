import logging
import os
import random
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import boto3
from boto3.exceptions import S3UploadFailedError
from boto3.s3.transfer import TransferConfig
from botocore.client import BaseClient
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError, ReadTimeoutError

RANGE_READ_WORKERS = 20
READ_BACKOFF_SECONDS = 0.1
READ_BACKOFF_MAX_SECONDS = 30.0
UPLOAD_ATTEMPTS = 3
UPLOAD_BACKOFF_SECONDS = 1.0
UPLOAD_PART_SIZE = 64 * 1024 * 1024
RETRYABLE_S3_CODES = frozenset(
    {"InternalError", "RequestTimeout", "ServiceUnavailable", "SlowDown"}
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ObjectStoreConfig:
    bucket: str
    folder: str
    endpoint_url: str
    region_name: str
    access_key_id: str
    secret_access_key: str

    @classmethod
    def from_env(cls) -> "ObjectStoreConfig":
        return cls(
            bucket=os.environ.get("RUNFLOW_S3_BUCKET", "runflow"),
            folder=os.environ.get("RUNFLOW_S3_FOLDER", "/"),
            endpoint_url=os.environ.get(
                "RUNFLOW_S3_ENDPOINT_URL", "http://127.0.0.1:9001"
            ),
            region_name=os.environ.get("RUNFLOW_S3_REGION", "us-east-1"),
            access_key_id=os.environ.get("RUNFLOW_S3_ACCESS_KEY_ID", "runflow"),
            secret_access_key=os.environ.get(
                "RUNFLOW_S3_SECRET_ACCESS_KEY", "runflow-secret"
            ),
        )


@dataclass(frozen=True)
class ObjectRange:
    path: str
    byte_offset: int
    byte_length: int


class S3RequestMetrics:
    def __init__(self) -> None:
        self.request_count = 0
        self.error_count = 0
        self.response_seconds = 0.0
        self.fetch_seconds = 0.0
        self.fetch_bytes = 0
        self.failed_queries = 0
        self.failed_query_codes: dict[str, int] = {}
        self._lock = threading.Lock()

    def before_call(self, context, **_kwargs) -> None:
        context["runflow_request_started"] = time.monotonic()

    def after_call(self, context, http_response, **_kwargs) -> None:
        self._record(context, http_response.status_code >= 300)

    def after_call_error(self, context, **_kwargs) -> None:
        self._record(context, True)

    def _record(self, context, failed: bool) -> None:
        started = context.get("runflow_request_started", time.monotonic())
        with self._lock:
            self.request_count += 1
            self.error_count += int(failed)
            self.response_seconds += time.monotonic() - started

    def record_failed_query(self, code: str) -> None:
        with self._lock:
            self.failed_queries += 1
            self.failed_query_codes[code] = (
                self.failed_query_codes.get(code, 0) + 1
            )


class ObjectStore(ABC):
    @abstractmethod
    def upload(self, path: str, data: bytes) -> None:
        raise NotImplementedError

    @abstractmethod
    def upload_path(self, path: str, source: Path) -> None:
        raise NotImplementedError

    @abstractmethod
    def test_connection(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def exists(self, path: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def download(self, path: str) -> bytes:
        raise NotImplementedError

    @abstractmethod
    def read_range(self, request: ObjectRange) -> bytes:
        raise NotImplementedError

    @abstractmethod
    def read_ranges(self, requests: Sequence[ObjectRange]) -> list[bytes]:
        raise NotImplementedError

    @abstractmethod
    def delete(self, path: str) -> None:
        raise NotImplementedError


class S3ObjectStore(ObjectStore):
    def __init__(
        self,
        config: ObjectStoreConfig,
        request_metrics: S3RequestMetrics | None = None,
    ) -> None:
        self._request_metrics = request_metrics
        self._bucket = config.bucket
        self._folder = _normalize_folder(config.folder)
        self._upload_config = TransferConfig(
            multipart_chunksize=UPLOAD_PART_SIZE,
            max_concurrency=1,
            use_threads=False,
        )
        self._client: BaseClient = boto3.client(
            "s3",
            endpoint_url=config.endpoint_url,
            region_name=config.region_name,
            aws_access_key_id=config.access_key_id,
            aws_secret_access_key=config.secret_access_key,
            config=Config(
                response_checksum_validation="when_required",
                max_pool_connections=RANGE_READ_WORKERS,
                read_timeout=1.5,
            ),
        )
        if request_metrics is not None:
            events = self._client.meta.events
            events.register("before-call.s3.GetObject", request_metrics.before_call)
            events.register("after-call.s3.GetObject", request_metrics.after_call)
            events.register(
                "after-call-error.s3.GetObject",
                request_metrics.after_call_error,
            )

    def upload(self, path: str, data: bytes) -> None:
        self._client.put_object(Bucket=self._bucket, Key=self._key(path), Body=data)

    def upload_path(self, path: str, source: Path) -> None:
        key = self._key(path)
        for attempt in range(UPLOAD_ATTEMPTS):
            try:
                self._client.upload_file(
                    str(source),
                    self._bucket,
                    key,
                    Config=self._upload_config,
                )
                return
            except S3UploadFailedError:
                if attempt == UPLOAD_ATTEMPTS - 1:
                    raise
                delay = UPLOAD_BACKOFF_SECONDS * (2**attempt)
                logger.warning(
                    "S3 upload failed path=%s attempt=%s/%s; retrying in %.2fs",
                    path,
                    attempt + 1,
                    UPLOAD_ATTEMPTS,
                    delay,
                    exc_info=True,
                )
                time.sleep(delay)
        raise AssertionError("file upload retry loop exhausted")

    def test_connection(self) -> None:
        self._client.head_bucket(Bucket=self._bucket)

    def exists(self, path: str) -> bool:
        try:
            self._client.head_object(Bucket=self._bucket, Key=self._key(path))
            return True
        except ClientError as error:
            code = error.response["Error"]["Code"]
            if code in {"404", "NoSuchKey", "NotFound"}:
                return False
            raise

    def download(self, path: str) -> bytes:
        return self._read(path, None, None)

    def read_range(self, request: ObjectRange) -> bytes:
        last_byte = request.byte_offset + request.byte_length - 1
        byte_range = f"bytes={request.byte_offset}-{last_byte}"
        return self._read(request.path, byte_range, request.byte_length)

    def _read(self, path: str, byte_range: str | None, length: int | None) -> bytes:
        attempt = 0
        while True:
            try:
                if byte_range is None:
                    response = self._client.get_object(
                        Bucket=self._bucket,
                        Key=self._key(path),
                    )
                else:
                    response = self._client.get_object(
                        Bucket=self._bucket, Key=self._key(path), Range=byte_range
                    )
                data = response["Body"].read()
                expected = int(response["ContentLength"]) if length is None else length
                if len(data) != expected:
                    raise EOFError(
                        f"{path} returned {len(data)} bytes; expected {expected}"
                    )
                return data
            except (BotoCoreError, EOFError) as error:
                if not _retryable_read_error(error):
                    raise
                self._record_read_failure(error)
                delay = _read_retry_delay(attempt)
                logger.warning(
                    "S3 read failed path=%s code=%s attempt=%s; retrying in %.2fs",
                    path,
                    _read_error_code(error),
                    attempt + 1,
                    delay,
                )
                time.sleep(delay)
                attempt += 1

    def read_ranges(self, requests: Sequence[ObjectRange]) -> list[bytes]:
        if not requests:
            return []
        with ThreadPoolExecutor(
            max_workers=min(RANGE_READ_WORKERS, len(requests)),
            thread_name_prefix="object-range",
        ) as executor:
            return list(executor.map(self.read_range, requests))

    def delete(self, path: str) -> None:
        self._client.delete_object(Bucket=self._bucket, Key=self._key(path))

    def _key(self, path: str) -> str:
        normalized_path = path.lstrip("/")
        if self._folder == "":
            return normalized_path
        return f"{self._folder}/{normalized_path}"

    def _record_read_failure(self, error: BotoCoreError | EOFError) -> None:
        if self._request_metrics is not None:
            self._request_metrics.record_failed_query(_read_error_code(error))


def _normalize_folder(folder: str) -> str:
    stripped = folder.strip().strip("/")
    return "" if stripped in {"", "."} else stripped


def _retryable_read_error(error: BotoCoreError | EOFError) -> bool:
    if isinstance(error, (ReadTimeoutError, EOFError)):
        return True
    if not isinstance(error, ClientError):
        return True
    response = error.response
    code = response["Error"]["Code"]
    status = int(response["ResponseMetadata"]["HTTPStatusCode"])
    return code in RETRYABLE_S3_CODES or status >= 500


def _read_error_code(error: BotoCoreError | EOFError) -> str:
    if isinstance(error, ClientError):
        return str(error.response["Error"]["Code"])
    return type(error).__name__


def _read_retry_delay(attempt: int) -> float:
    exponential_delay = READ_BACKOFF_SECONDS * (2 ** min(attempt, 20))
    capped_delay = min(exponential_delay, READ_BACKOFF_MAX_SECONDS)
    return capped_delay * random.uniform(0.75, 1.25)

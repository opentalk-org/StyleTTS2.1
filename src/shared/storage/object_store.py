import os
from abc import ABC, abstractmethod
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import boto3
from botocore.client import BaseClient
from botocore.config import Config
from botocore.exceptions import ClientError, ReadTimeoutError

RANGE_READ_ATTEMPTS = 3
RANGE_READ_WORKERS = 10


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
                "RUNFLOW_S3_ENDPOINT_URL", "http://127.0.0.1:9000"
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
    def __init__(self, config: ObjectStoreConfig) -> None:
        self._bucket = config.bucket
        self._folder = _normalize_folder(config.folder)
        self._client: BaseClient = boto3.client(
            "s3",
            endpoint_url=config.endpoint_url,
            region_name=config.region_name,
            aws_access_key_id=config.access_key_id,
            aws_secret_access_key=config.secret_access_key,
            config=Config(response_checksum_validation="when_required"),
        )

    def upload(self, path: str, data: bytes) -> None:
        self._client.put_object(Bucket=self._bucket, Key=self._key(path), Body=data)

    def upload_path(self, path: str, source: Path) -> None:
        self._client.upload_file(str(source), self._bucket, self._key(path))

    def test_connection(self) -> None:
        self._client.head_bucket(Bucket=self._bucket)

    def download(self, path: str) -> bytes:
        response = self._client.get_object(Bucket=self._bucket, Key=self._key(path))
        return response["Body"].read()

    def read_range(self, request: ObjectRange) -> bytes:
        last_byte = request.byte_offset + request.byte_length - 1
        for attempt in range(RANGE_READ_ATTEMPTS):
            try:
                response = self._client.get_object(
                    Bucket=self._bucket,
                    Key=self._key(request.path),
                    Range=f"bytes={request.byte_offset}-{last_byte}",
                )
                data = response["Body"].read()
                break
            except (ClientError, ReadTimeoutError) as error:
                retryable = (
                    isinstance(error, ReadTimeoutError)
                    or error.response["Error"]["Code"] == "NoSuchKey"
                )
                if not retryable or attempt == RANGE_READ_ATTEMPTS - 1:
                    raise
        else:
            raise AssertionError(
                "range read retry loop exhausted without returning or raising"
            )
        if len(data) != request.byte_length:
            raise EOFError(
                f"{request.path} returned {len(data)} bytes; "
                f"expected {request.byte_length}"
            )
        return data

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


def _normalize_folder(folder: str) -> str:
    stripped = folder.strip().strip("/")
    if stripped in {"", "."}:
        return ""
    return stripped

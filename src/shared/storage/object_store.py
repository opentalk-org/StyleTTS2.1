from dataclasses import dataclass
import os

import boto3
from botocore.client import BaseClient


@dataclass(frozen=True)
class ObjectStoreConfig:
    bucket: str
    endpoint_url: str
    region_name: str
    access_key_id: str
    secret_access_key: str

    @classmethod
    def from_env(cls) -> "ObjectStoreConfig":
        return cls(
            bucket=os.environ.get("RUNFLOW_S3_BUCKET", "runflow"),
            endpoint_url=os.environ.get("RUNFLOW_S3_ENDPOINT_URL", "http://127.0.0.1:9000"),
            region_name=os.environ.get("RUNFLOW_S3_REGION", "us-east-1"),
            access_key_id=os.environ.get("RUNFLOW_S3_ACCESS_KEY_ID", "runflow"),
            secret_access_key=os.environ.get("RUNFLOW_S3_SECRET_ACCESS_KEY", "runflow-secret"),
        )


class S3ObjectStore:
    def __init__(self, config: ObjectStoreConfig) -> None:
        self._bucket = config.bucket
        self._client: BaseClient = boto3.client(
            "s3",
            endpoint_url=config.endpoint_url,
            region_name=config.region_name,
            aws_access_key_id=config.access_key_id,
            aws_secret_access_key=config.secret_access_key,
        )

    def upload(self, path: str, data: bytes) -> None:
        self._client.put_object(Bucket=self._bucket, Key=path, Body=data)

    def download(self, path: str) -> bytes:
        response = self._client.get_object(Bucket=self._bucket, Key=path)
        return response["Body"].read()

    def read_range(self, path: str, byte_offset: int, byte_length: int) -> bytes:
        last_byte = byte_offset + byte_length - 1
        response = self._client.get_object(
            Bucket=self._bucket,
            Key=path,
            Range=f"bytes={byte_offset}-{last_byte}",
        )
        return response["Body"].read()

    def delete(self, path: str) -> None:
        self._client.delete_object(Bucket=self._bucket, Key=path)

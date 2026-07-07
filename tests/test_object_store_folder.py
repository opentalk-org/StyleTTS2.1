from __future__ import annotations

import io
import unittest
from unittest.mock import patch

from shared.storage.object_store import ObjectStoreConfig, S3ObjectStore


class ObjectStoreFolderTests(unittest.TestCase):
    def test_root_folder_keeps_existing_object_keys(self) -> None:
        client = _FakeS3Client()
        store = _store_with_client(client, folder="/")

        store.upload("audio/pack.bin", b"data")
        store.download("audio/pack.bin")
        store.read_range("audio/pack.bin", 2, 3)
        store.delete("audio/pack.bin")

        self.assertEqual(client.calls, [
            ("put_object", "audio/pack.bin", None),
            ("get_object", "audio/pack.bin", None),
            ("get_object", "audio/pack.bin", "bytes=2-4"),
            ("delete_object", "audio/pack.bin", None),
        ])

    def test_folder_prefixes_object_keys(self) -> None:
        client = _FakeS3Client()
        store = _store_with_client(client, folder="tenant-a")

        store.upload("/audio/pack.bin", b"data")
        store.download("audio/pack.bin")
        store.delete("audio/pack.bin")

        self.assertEqual(client.calls, [
            ("put_object", "tenant-a/audio/pack.bin", None),
            ("get_object", "tenant-a/audio/pack.bin", None),
            ("delete_object", "tenant-a/audio/pack.bin", None),
        ])

    def test_connection_checks_bucket_without_writing_objects(self) -> None:
        client = _FakeS3Client()
        store = _store_with_client(client, folder="tenant-a")

        store.test_connection()

        self.assertEqual(client.calls, [("head_bucket", "runflow", None)])


class _FakeS3Client:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str | None]] = []

    def put_object(self, *, Bucket: str, Key: str, Body: bytes) -> None:
        self.calls.append(("put_object", Key, None))

    def get_object(self, *, Bucket: str, Key: str, Range: str | None = None) -> dict:
        self.calls.append(("get_object", Key, Range))
        return {"Body": io.BytesIO(b"data")}

    def delete_object(self, *, Bucket: str, Key: str) -> None:
        self.calls.append(("delete_object", Key, None))

    def head_bucket(self, *, Bucket: str) -> None:
        self.calls.append(("head_bucket", Bucket, None))


def _store_with_client(client: _FakeS3Client, *, folder: str) -> S3ObjectStore:
    config = ObjectStoreConfig(
        bucket="runflow",
        folder=folder,
        endpoint_url="http://127.0.0.1:9000",
        region_name="us-east-1",
        access_key_id="runflow",
        secret_access_key="runflow-secret",
    )
    with patch("shared.storage.object_store.boto3.client", return_value=client):
        return S3ObjectStore(config)


if __name__ == "__main__":
    unittest.main()

import hashlib


def derive_seed(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    digest = int.from_bytes(
        hashlib.sha256(payload).digest()[:8],
        "big",
        signed=False,
    )
    return digest & ((1 << 63) - 1)

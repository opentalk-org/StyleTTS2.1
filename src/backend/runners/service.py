from datetime import UTC, datetime


STALE_AFTER_SECONDS = 15


def runner_is_online(last_seen_at: datetime | None) -> bool:
    if last_seen_at is None:
        return False
    return (datetime.now(UTC) - last_seen_at).total_seconds() <= STALE_AFTER_SECONDS

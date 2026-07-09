"""Merge per-node run logs into one timestamp-ordered, node-tagged stream.

This is the Python port of the merge the frontend performs in
``AllLogsPopover.tsx`` ("All node logs (merged by time)"). Keeping the two in
sync means the aggregated logs an operator reads in the UI and the ones this CLI
prints are byte-for-byte the same.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Node log lines are prefixed by "YYYY-MM-DD HH:MM:SS,mmm" (comma or dot millis),
# written by ``CappedNodeLogHandler`` in ``runner/node_logs.py``.
_TIMESTAMP = re.compile(r"^(\d{4}-\d{2}-\d{2})[ T](\d{2}:\d{2}:\d{2})[.,](\d{3})")


@dataclass(frozen=True)
class NodeLog:
    """One node's raw log, as returned by the per-node log endpoint."""

    node_id: str
    content: str
    truncated: bool = False
    error: str | None = None


@dataclass(frozen=True)
class LogRecord:
    node_id: str
    text: str
    ts: float | None
    seq: int


def _parse_timestamp(line: str) -> float | None:
    match = _TIMESTAMP.match(line)
    if match is None:
        return None
    # A lexical key is enough for ordering and avoids datetime parsing costs.
    return _lexical_key(match.group(1), match.group(2), match.group(3))


def _lexical_key(date: str, time: str, millis: str) -> float:
    digits = f"{date.replace('-', '')}{time.replace(':', '')}{millis}"
    return float(digits)


def _to_records(log: NodeLog, seq_start: int) -> list[LogRecord]:
    """Split one node's log into records.

    A record starts at a timestamped line and absorbs the following continuation
    lines (e.g. tracebacks) so multi-line entries stay together.
    """
    records: list[LogRecord] = []
    seq = seq_start
    if log.error:
        records.append(LogRecord(log.node_id, f"[error] {log.error}", None, seq))
        seq += 1
    current: LogRecord | None = None
    for line in log.content.split("\n"):
        if line == "" and current is None:
            continue
        ts = _parse_timestamp(line)
        if ts is not None or current is None:
            current = LogRecord(log.node_id, line, ts, seq)
            records.append(current)
            seq += 1
        else:
            current = LogRecord(current.node_id, f"{current.text}\n{line}", current.ts, current.seq)
            records[-1] = current
    return records


def merge_logs(logs: list[NodeLog]) -> list[LogRecord]:
    """Merge node logs into one chronological stream.

    Lines without a timestamp keep their relative order (stable sort on the
    per-record sequence number, exactly like the UI).
    """
    all_records: list[LogRecord] = []
    seq = 0
    for log in logs:
        records = _to_records(log, seq)
        seq += len(records)
        all_records.extend(records)
    all_records.sort(key=lambda record: (record.ts if record.ts is not None else float("-inf"), record.seq))
    return all_records


def format_merged(logs: list[NodeLog]) -> str:
    """Render merged logs as ``"{node_id}\\t{text}"`` lines (the UI 'Copy all' format)."""
    return "\n".join(f"{record.node_id}\t{record.text}" for record in merge_logs(logs))

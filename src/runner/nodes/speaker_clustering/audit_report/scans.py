from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


PROJECTED_ASSIGNMENT_SCHEMA = pa.schema(
    [
        pa.field("segment_id", pa.string(), nullable=False),
        pa.field("audio_id", pa.string(), nullable=False),
        pa.field("duration_seconds", pa.float32(), nullable=False),
        pa.field("cluster_id", pa.int64()),
        pa.field("best_cluster_id", pa.int64()),
        pa.field("second_cluster_id", pa.int64()),
        pa.field("best_score", pa.float32()),
        pa.field("second_score", pa.float32()),
        pa.field("margin", pa.float32()),
        pa.field("candidate_scores", pa.list_(pa.float32()), nullable=False),
        pa.field("true_label", pa.string()),
    ]
)


@dataclass(frozen=True)
class AssignmentParquetScan:
    paths: tuple[Path, ...]
    batch_rows: int
    total_rows: int
    check_cancel: Callable[[], None]
    report_progress: Callable[[int, int], None]

    @classmethod
    def create(
        cls,
        paths: Sequence[Path],
        batch_rows: int,
        required_columns: frozenset[str],
        check_cancel: Callable[[], None],
        report_progress: Callable[[int, int], None],
    ) -> AssignmentParquetScan:
        if batch_rows <= 0:
            raise ValueError("batch_rows must be positive")
        resolved = tuple(Path(path) for path in paths)
        total_rows = 0
        for path in resolved:
            parquet = pq.ParquetFile(path)
            missing = sorted(required_columns.difference(parquet.schema_arrow.names))
            if missing:
                raise ValueError(
                    f"assignment parquet {path} is missing columns: {', '.join(missing)}"
                )
            incompatible = sorted(
                field.name
                for field in PROJECTED_ASSIGNMENT_SCHEMA
                if field.name in required_columns
                and not _compatible_field(parquet.schema_arrow.field(field.name), field)
            )
            if incompatible:
                raise ValueError(
                    f"assignment parquet {path} has incompatible columns: "
                    f"{', '.join(incompatible)}"
                )
            total_rows += parquet.metadata.num_rows
        return cls(
            paths=resolved,
            batch_rows=batch_rows,
            total_rows=total_rows,
            check_cancel=check_cancel,
            report_progress=report_progress,
        )

    def batches(self, columns: Sequence[str]) -> Iterator[pa.RecordBatch]:
        processed = 0
        if self.total_rows == 0:
            self.report_progress(0, 0)
            return
        for path in self.paths:
            parquet = pq.ParquetFile(path)
            for batch in parquet.iter_batches(
                batch_size=self.batch_rows,
                columns=columns,
            ):
                self.check_cancel()
                yield batch
                processed += batch.num_rows
                self.report_progress(processed, self.total_rows)


def _compatible_field(actual: pa.Field, expected: pa.Field) -> bool:
    nullable_is_compatible = expected.nullable or not actual.nullable
    return actual.type == expected.type and nullable_is_compatible

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from runner.nodes.speaker_clustering.candidates import (
    CandidateMatrix,
    ReciprocalCandidateLookup,
)


EDGE_SCHEMA = pa.schema(
    [
        pa.field("left_id", pa.int64(), nullable=False),
        pa.field("right_id", pa.int64(), nullable=False),
        pa.field("exact_score", pa.float32(), nullable=False),
        pa.field("reciprocal_rank", pa.int16(), nullable=False),
    ]
)


@dataclass(frozen=True)
class EdgeBlock:
    left_ids: np.ndarray
    right_ids: np.ndarray
    exact_scores: np.ndarray
    reciprocal_ranks: np.ndarray


def iter_reciprocal_edge_blocks(
    candidates: CandidateMatrix,
    block_rows: int,
    check_cancel: Callable[[], None] | None = None,
) -> Iterator[EdgeBlock]:
    if block_rows <= 0:
        raise ValueError(f"edge block_rows must be positive, got {block_rows}")
    with TemporaryDirectory(prefix="reciprocal-candidates-") as directory:
        lookup = ReciprocalCandidateLookup.create(
            Path(directory) / "lookup.sqlite3",
            candidates,
            block_rows,
            check_cancel,
        )
        try:
            for start in range(0, candidates.item_count, block_rows):
                if check_cancel is not None:
                    check_cancel()
                stop = min(start + block_rows, candidates.item_count)
                forward_ids = np.asarray(candidates.row_ids[start:stop])
                forward_scores = np.asarray(candidates.scores[start:stop])
                left_values: list[int] = []
                right_values: list[int] = []
                score_values: list[float] = []
                rank_values: list[int] = []
                for row_offset in range(stop - start):
                    if check_cancel is not None:
                        check_cancel()
                    left_id = start + row_offset
                    for forward_rank in range(candidates.neighbors):
                        right_id = int(forward_ids[row_offset, forward_rank])
                        if right_id < 0 or left_id >= right_id:
                            continue
                        reverse = lookup.get(right_id, left_id)
                        if reverse is None:
                            continue
                        left_values.append(left_id)
                        right_values.append(right_id)
                        score_values.append(
                            min(
                                float(forward_scores[row_offset, forward_rank]),
                                reverse.score,
                            )
                        )
                        rank_values.append(reverse.rank)
                if not left_values:
                    continue
                block = EdgeBlock(
                    left_ids=np.asarray(left_values, dtype=np.int64),
                    right_ids=np.asarray(right_values, dtype=np.int64),
                    exact_scores=np.asarray(score_values, dtype=np.float32),
                    reciprocal_ranks=np.asarray(rank_values, dtype=np.int16),
                )
                yield _sorted_edges(block)
        finally:
            lookup.close()


def write_reciprocal_edge_shards(
    candidates: CandidateMatrix,
    output_dir: Path,
    block_rows: int,
    shard_rows: int,
    check_cancel: Callable[[], None] | None = None,
) -> list[Path]:
    if shard_rows <= 0:
        raise ValueError(f"edge shard_rows must be positive, got {shard_rows}")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    ordinal = 0
    for block in iter_reciprocal_edge_blocks(candidates, block_rows, check_cancel):
        for start in range(0, len(block.left_ids), shard_rows):
            if check_cancel is not None:
                check_cancel()
            stop = min(start + shard_rows, len(block.left_ids))
            path = output_dir / f"edges-{ordinal:08d}.parquet"
            pq.write_table(_edge_table(block, start, stop), path, compression="zstd")
            paths.append(path)
            ordinal += 1
    return paths


def _edge_table(block: EdgeBlock, start: int, stop: int) -> pa.Table:
    return pa.Table.from_arrays(
        [
            pa.array(block.left_ids[start:stop], type=pa.int64()),
            pa.array(block.right_ids[start:stop], type=pa.int64()),
            pa.array(block.exact_scores[start:stop], type=pa.float32()),
            pa.array(block.reciprocal_ranks[start:stop], type=pa.int16()),
        ],
        schema=EDGE_SCHEMA,
    )


def _sorted_edges(block: EdgeBlock) -> EdgeBlock:
    order = np.lexsort((block.right_ids, block.left_ids))
    return EdgeBlock(
        left_ids=block.left_ids[order],
        right_ids=block.right_ids[order],
        exact_scores=block.exact_scores[order],
        reciprocal_ranks=block.reciprocal_ranks[order],
    )

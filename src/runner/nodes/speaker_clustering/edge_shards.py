from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from runner.nodes.speaker_clustering.candidates import CandidateMatrix
from runner.nodes.speaker_clustering.reciprocal_index import (
    RowSortedCandidateIndex,
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
    index: RowSortedCandidateIndex,
    block_rows: int,
    check_cancel: Callable[[], None] | None = None,
) -> Iterator[EdgeBlock]:
    if block_rows <= 0:
        raise ValueError(f"edge block_rows must be positive, got {block_rows}")
    for start in range(0, candidates.item_count, block_rows):
        if check_cancel is not None:
            check_cancel()
        stop = min(start + block_rows, candidates.item_count)
        left_ids = np.arange(start, stop, dtype=np.int64)
        forward_ids = np.asarray(candidates.row_ids[start:stop])
        forward_scores = np.asarray(candidates.scores[start:stop])
        left_grid = np.broadcast_to(left_ids[:, np.newaxis], forward_ids.shape)
        reverse = index.search(forward_ids, left_grid, check_cancel)
        accepted = reverse.found & (left_grid < forward_ids)
        if not np.any(accepted):
            continue
        block = EdgeBlock(
            left_ids=left_grid[accepted],
            right_ids=forward_ids[accepted],
            exact_scores=np.minimum(forward_scores, reverse.scores)[accepted],
            reciprocal_ranks=reverse.ranks[accepted].astype(np.int16),
        )
        yield _sorted_edges(block)


def write_reciprocal_edge_shards(
    candidates: CandidateMatrix,
    output_dir: Path,
    block_rows: int,
    shard_rows: int,
    scratch_root: Path,
    check_cancel: Callable[[], None] | None = None,
) -> list[Path]:
    if shard_rows <= 0:
        raise ValueError(f"edge shard_rows must be positive, got {shard_rows}")
    output_dir.mkdir(parents=True, exist_ok=True)
    scratch_root.mkdir(parents=True, exist_ok=True)
    paths = []
    ordinal = 0
    with TemporaryDirectory(prefix="reciprocal-candidates-", dir=scratch_root) as work:
        index = RowSortedCandidateIndex.create(
            Path(work) / "index", candidates, block_rows, check_cancel
        )
        try:
            blocks = iter_reciprocal_edge_blocks(
                candidates, index, block_rows, check_cancel
            )
            for block in blocks:
                for start in range(0, len(block.left_ids), shard_rows):
                    if check_cancel is not None:
                        check_cancel()
                    stop = min(start + shard_rows, len(block.left_ids))
                    path = output_dir / f"edges-{ordinal:08d}.parquet"
                    pq.write_table(
                        _edge_table(block, start, stop), path, compression="zstd"
                    )
                    paths.append(path)
                    ordinal += 1
        finally:
            index.close()
    return paths


def iter_edge_paths(
    paths: Sequence[Path],
    batch_rows: int,
    check_cancel: Callable[[], None] | None = None,
) -> Iterator[EdgeBlock]:
    if batch_rows <= 0:
        raise ValueError(f"edge batch_rows must be positive, got {batch_rows}")
    for path in paths:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=batch_rows):
            if check_cancel is not None:
                check_cancel()
            yield EdgeBlock(
                left_ids=np.asarray(batch.column("left_id"), dtype=np.int64),
                right_ids=np.asarray(batch.column("right_id"), dtype=np.int64),
                exact_scores=np.asarray(batch.column("exact_score"), dtype=np.float32),
                reciprocal_ranks=np.asarray(
                    batch.column("reciprocal_rank"), dtype=np.int16
                ),
            )


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

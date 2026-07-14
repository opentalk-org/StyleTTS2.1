from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from runner.nodes.speaker_clustering.candidates import CandidateMatrix


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
) -> Iterator[EdgeBlock]:
    if block_rows <= 0:
        raise ValueError(f"edge block_rows must be positive, got {block_rows}")
    for start in range(0, candidates.item_count, block_rows):
        stop = min(start + block_rows, candidates.item_count)
        left_ids = np.arange(start, stop, dtype=np.int64)
        forward_ids = np.asarray(candidates.row_ids[start:stop])
        forward_scores = np.asarray(candidates.scores[start:stop])
        valid = forward_ids >= 0
        safe_ids = np.where(valid, forward_ids, 0)
        reverse_ids = np.asarray(candidates.row_ids[safe_ids])
        reverse_scores = np.asarray(candidates.scores[safe_ids])
        matches = reverse_ids == left_ids[:, np.newaxis, np.newaxis]
        reciprocal = np.any(matches, axis=2)
        reverse_positions = np.argmax(matches, axis=2)
        reverse_exact = np.take_along_axis(
            reverse_scores,
            reverse_positions[:, :, np.newaxis],
            axis=2,
        ).squeeze(axis=2)
        accepted = valid & reciprocal & (left_ids[:, np.newaxis] < forward_ids)
        if not np.any(accepted):
            continue
        left_grid = np.broadcast_to(left_ids[:, np.newaxis], forward_ids.shape)
        block = EdgeBlock(
            left_ids=left_grid[accepted],
            right_ids=forward_ids[accepted],
            exact_scores=np.minimum(forward_scores, reverse_exact)[accepted],
            reciprocal_ranks=(reverse_positions[accepted] + 1).astype(np.int16),
        )
        yield _sorted_edges(block)


def write_reciprocal_edge_shards(
    candidates: CandidateMatrix,
    output_dir: Path,
    block_rows: int,
    shard_rows: int,
) -> list[Path]:
    if shard_rows <= 0:
        raise ValueError(f"edge shard_rows must be positive, got {shard_rows}")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    ordinal = 0
    for block in iter_reciprocal_edge_blocks(candidates, block_rows):
        for start in range(0, len(block.left_ids), shard_rows):
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

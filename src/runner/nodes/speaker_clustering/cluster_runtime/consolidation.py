from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
import shutil

import numpy as np

from runner.nodes.models import SpeakerEmbeddingSetRef
from runner.nodes.speaker_clustering.cluster_runtime.assignment_runtime import (
    prototype_neighbor_ids,
)
from runner.nodes.speaker_clustering.edge_shards import EdgeBlock
from runner.nodes.speaker_clustering.prototypes import (
    build_prototype_store,
    consolidate_labels_on_disk,
)
from runner.nodes.speaker_clustering.shard_reader import EmbeddingBlock


def consolidate_hierarchically(
    blocks: Callable[[], Iterable[EmbeddingBlock]],
    edges: Callable[[], Iterable[EdgeBlock]],
    labels: np.ndarray,
    embedding_set: SpeakerEmbeddingSetRef,
    settings: object,
    scratch: Path,
    check_cancel: Callable[[], None],
) -> None:
    prototype_dir = scratch / "seed-prototypes"
    support_path = scratch / "support.sqlite3"
    for _round in range(settings.max_consolidation_rounds):
        check_cancel()
        prototypes = build_prototype_store(
            block_factory=blocks,
            labels=labels,
            directory=prototype_dir,
            item_count=embedding_set.item_count,
            dimension=embedding_set.dimension,
            max_members=settings.max_cluster_members,
            max_dispersion=settings.max_cluster_dispersion,
            block_rows=settings.block_rows,
            check_cancel=check_cancel,
        )
        prototype_neighbors = prototype_neighbor_ids(
            prototypes,
            settings,
            prototype_dir / "neighbors.i64",
            prototype_dir / "neighbor-established.bool",
            check_cancel,
        )
        support_path.unlink(missing_ok=True)
        try:
            merged_count = consolidate_labels_on_disk(
                labels=labels,
                edge_blocks=edges(),
                database_path=support_path,
                min_support_pairs=settings.min_support_pairs,
                max_members=settings.max_cluster_members,
                block_rows=settings.block_rows,
                check_cancel=check_cancel,
                prototype_neighbors=prototype_neighbors,
            )
        finally:
            prototype_neighbors._mmap.close()
        del prototypes
        shutil.rmtree(prototype_dir)
        support_path.unlink(missing_ok=True)
        if merged_count == 0:
            return

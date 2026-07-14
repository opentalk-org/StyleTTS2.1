from __future__ import annotations

from collections.abc import Callable, Iterable
from functools import partial
from pathlib import Path
import shutil
from uuid import UUID

import numpy as np
from pydantic import Field

from runflow.core.settings import StrictSettings
from runner.nodes.models import SpeakerClusterRunRef, SpeakerEmbeddingSetRef
from runner.nodes.speaker_clustering.candidates import (
    CandidateMatrix,
    build_canonical_store,
    rerank_candidate_blocks,
)
from runner.nodes.speaker_clustering.cluster_runtime.assignment_runtime import (
    assignment_blocks,
    build_prototype_index,
    prototype_neighbor_ids,
)
from runner.nodes.speaker_clustering.cluster_runtime.artifacts import (
    write_assignment_shards,
    write_prototype_shards,
)
from runner.nodes.speaker_clustering.cluster_runtime.persistence import (
    fail_clustering_run,
    persist_clustering_outputs,
    prepare_clustering_run,
)
from shared.db.speakers.schemas import ClusteringOutcomeCounts
from runner.nodes.speaker_clustering.edge_shards import (
    EdgeBlock,
    iter_edge_paths,
    write_reciprocal_edge_shards,
)
from runner.nodes.speaker_clustering.faiss_index import FaissIndexSettings
from runner.nodes.speaker_clustering.faiss_index import (
    build_candidate_index_from_blocks,
)
from runner.nodes.speaker_clustering.microclusters import build_microcluster_labels
from runner.nodes.speaker_clustering.prototypes import build_prototype_store
from runner.nodes.speaker_clustering.prototypes import consolidate_labels_on_disk
from runner.nodes.speaker_clustering.shard_reader import (
    EmbeddingBlock,
    iter_embedding_blocks,
)


class ClusterSpeakerEmbeddingsSettings(StrictSettings):
    index_factory: str = "IVF65536_HNSW32,Flat"
    training_rows: int = Field(default=1_000_000, gt=0)
    search_probes: int = Field(default=64, gt=0)
    random_seed: int = 0
    block_rows: int = Field(default=4096, gt=0)
    candidate_neighbors: int = Field(default=64, gt=1)
    exact_edge_threshold: float = Field(default=0.65, ge=-1.0, le=1.0)
    edge_shard_rows: int = Field(default=500_000, gt=0)
    min_support_pairs: int = Field(default=2, gt=1)
    max_consolidation_rounds: int = Field(default=12, gt=0)
    max_cluster_members: int = Field(default=10_000, gt=1)
    max_cluster_dispersion: float = Field(default=0.20, ge=0.0, le=2.0)
    prototype_min_members: int = Field(default=2, gt=1)
    prototype_index_factory: str = "HNSW32"
    prototype_search_effort: int = Field(default=64, gt=0)
    assignment_neighbors: int = Field(default=8, gt=1)
    accept_threshold: float = Field(default=0.70, ge=-1.0, le=1.0)
    min_margin: float = Field(default=0.10, ge=0.0, le=2.0)
    new_threshold: float = Field(default=0.45, ge=-1.0, le=1.0)
    dispersion_penalty: float = Field(default=0.25, ge=0.0)
    assignment_shard_rows: int = Field(default=100_000, gt=0)
    prototype_shard_rows: int = Field(default=50_000, gt=0)
    threshold_version: str = "ecapa-voxceleb-conservative-v1"

    def model_post_init(self, __context: object) -> None:
        if self.new_threshold >= self.accept_threshold:
            raise ValueError("new_threshold must be lower than accept_threshold")


def run_clustering_pipeline(
    embedding_set: SpeakerEmbeddingSetRef,
    settings: ClusterSpeakerEmbeddingsSettings,
    work_dir: Path,
    execution_run_id: str,
    node_id: str,
    check_cancel: Callable[[], None],
    report_stage: Callable[[int, str], None],
) -> SpeakerClusterRunRef:
    run_id, completed = prepare_clustering_run(
        embedding_set,
        execution_run_id,
        node_id,
        settings.index_factory,
        settings.threshold_version,
        settings.model_dump(mode="json"),
    )
    if completed is not None:
        return completed
    scratch = work_dir / "speaker-clustering"
    try:
        return _execute_clustering_pipeline(
            run_id, embedding_set, settings, work_dir, check_cancel, report_stage
        )
    except BaseException as error:
        fail_clustering_run(run_id, error)
        raise
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def _execute_clustering_pipeline(
    run_id: UUID, embedding_set: SpeakerEmbeddingSetRef,
    settings: ClusterSpeakerEmbeddingsSettings, work_dir: Path,
    check_cancel: Callable[[], None],
    report_stage: Callable[[int, str], None],
) -> SpeakerClusterRunRef:
    scratch = work_dir / "speaker-clustering"
    shutil.rmtree(scratch, ignore_errors=True)
    scratch.mkdir(parents=True)
    blocks = partial(
        iter_embedding_blocks,
        embedding_set,
        settings.block_rows,
        check_cancel,
    )
    report_stage(1, "training and filling candidate index")
    index_settings = FaissIndexSettings(
        settings.index_factory,
        settings.training_rows,
        settings.search_probes,
        settings.random_seed,
    )
    index = build_candidate_index_from_blocks(
        blocks,
        embedding_set.dimension,
        index_settings,
        check_cancel,
    )
    index_path = scratch / "candidates.faiss"
    index.save(index_path)
    report_stage(2, "materializing canonical vectors")
    canonical = build_canonical_store(
        blocks(),
        scratch / "canonical",
        embedding_set.item_count,
        embedding_set.dimension,
        check_cancel,
    )
    report_stage(3, "exact-reranking ANN candidates")
    candidate_matrix = CandidateMatrix.create(
        scratch / "candidate-matrix",
        embedding_set.item_count,
        settings.candidate_neighbors,
    )
    rerank_candidate_blocks(
        index,
        blocks(),
        canonical,
        candidate_matrix,
        settings.exact_edge_threshold,
        check_cancel,
    )
    edge_paths = write_reciprocal_edge_shards(
        candidate_matrix,
        scratch / "edges",
        settings.block_rows,
        settings.edge_shard_rows,
        scratch / "edge-work",
        check_cancel,
    )
    edges = partial(iter_edge_paths, edge_paths, settings.block_rows, check_cancel)
    report_stage(4, "forming mutual-best microclusters")
    labels = build_microcluster_labels(
        item_count=embedding_set.item_count,
        accepted=canonical.accepted,
        edge_blocks=edges(),
        directory=scratch / "microclusters",
        block_rows=settings.block_rows,
        check_cancel=check_cancel,
    )
    report_stage(5, "consolidating clusters with multi-pair support")
    _consolidate_hierarchically(
        blocks,
        edges,
        labels.values,
        embedding_set,
        settings,
        scratch,
        check_cancel,
    )
    report_stage(6, "building prototypes and dispersion diagnostics")
    prototypes = build_prototype_store(
        block_factory=blocks,
        labels=labels.values,
        directory=scratch / "prototypes",
        item_count=embedding_set.item_count,
        dimension=embedding_set.dimension,
        max_members=settings.max_cluster_members,
        max_dispersion=settings.max_cluster_dispersion,
        block_rows=settings.block_rows,
        check_cancel=check_cancel,
    )
    report_stage(7, "assigning accepted, provisional, ambiguous, and rejected rows")
    prototype_index, established = build_prototype_index(prototypes, settings)
    assignment_result = write_assignment_shards(
        assignment_blocks(
            blocks(),
            labels.values,
            prototypes,
            prototype_index,
            established,
            settings,
            check_cancel,
        ),
        scratch / "assignments",
        settings.assignment_shard_rows,
    )
    assigned_count = (
        assignment_result.counts.accepted
        + assignment_result.counts.provisional_new
        + assignment_result.counts.ambiguous
        + assignment_result.counts.rejected
    )
    if assigned_count != embedding_set.item_count:
        raise ValueError(
            f"assignment output has {assigned_count} rows, expected {embedding_set.item_count}"
        )
    prototype_paths = write_prototype_shards(
        prototypes,
        scratch / "prototype-shards",
        settings.prototype_shard_rows,
        settings.block_rows,
    )
    report_stage(8, "uploading durable clustering artifacts")
    result = persist_clustering_outputs(
        run_id,
        assignment_result.paths,
        prototype_paths,
        index_path,
        embedding_set.item_count,
        ClusteringOutcomeCounts(**assignment_result.counts.__dict__),
    )
    report_stage(9, "speaker clustering complete")
    return result


def _consolidate_hierarchically(
    blocks: Callable[[], Iterable[EmbeddingBlock]],
    edges: Callable[[], Iterable[EdgeBlock]],
    labels: np.ndarray,
    embedding_set: SpeakerEmbeddingSetRef,
    settings: ClusterSpeakerEmbeddingsSettings,
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
        prototype_neighbors = prototype_neighbor_ids(prototypes, settings)
        support_path.unlink(missing_ok=True)
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
        del prototypes
        shutil.rmtree(prototype_dir)
        support_path.unlink(missing_ok=True)
        if merged_count == 0:
            return

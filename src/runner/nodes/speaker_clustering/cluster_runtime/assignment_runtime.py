from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from pathlib import Path

import numpy as np

from runner.nodes.speaker_clustering.candidates import exact_rerank
from runner.nodes.speaker_clustering.cluster_runtime.artifacts import AssignmentRow
from runner.nodes.speaker_clustering.cluster_runtime.assignment import (
    AssignmentDecision,
    AssignmentPolicy,
    CandidateScores,
    decide,
)
from runner.nodes.speaker_clustering.cluster_runtime.prototype_blocks import (
    PrototypeSelection,
    build_prototype_selection,
    create_prototype_neighbor_ids,
)
from runner.nodes.speaker_clustering.faiss_index import (
    FaissIndexSettings,
    SpeakerCandidateIndex,
)
from runner.nodes.speaker_clustering.cluster_runtime.prototypes import PrototypeStore
from runner.nodes.speaker_clustering.reservoir import DeterministicVectorReservoir
from runner.nodes.speaker_clustering.shard_reader import EmbeddingBlock
from runner.nodes.speaker_clustering.shards import EmbeddingQuality


def build_prototype_index(
    prototypes: PrototypeStore,
    settings: object,
    selection_path: Path,
    check_cancel: Callable[[], None],
) -> tuple[SpeakerCandidateIndex | None, PrototypeSelection]:
    selection = build_prototype_selection(
        prototypes.member_counts,
        prototypes.suspicious,
        selection_path,
        settings.prototype_min_members,
        settings.block_rows,
        check_cancel,
    )
    try:
        index = _build_index_for_selection(
            prototypes, selection, settings, check_cancel
        )
        return index, selection
    except BaseException:
        selection.close()
        raise


def prototype_neighbor_ids(
    prototypes: PrototypeStore,
    settings: object,
    neighbor_path: Path,
    selection_path: Path,
    check_cancel: Callable[[], None],
) -> np.memmap:
    selection = build_prototype_selection(
        prototypes.member_counts,
        prototypes.suspicious,
        selection_path,
        settings.min_support_pairs,
        settings.block_rows,
        check_cancel,
    )
    neighbors = create_prototype_neighbor_ids(
        neighbor_path,
        prototypes.item_count,
        settings.block_rows,
        check_cancel,
    )
    try:
        index = _build_index_for_selection(
            prototypes, selection, settings, check_cancel
        )
        if index is None or index.item_count < 2:
            return neighbors
        search_count = min(settings.assignment_neighbors, index.item_count)
        for row_ids, vectors in _prototype_blocks(
            prototypes, selection, settings.block_rows, check_cancel
        ):
            check_cancel()
            proposed = index.search(vectors, search_count)
            check_cancel()
            reranked = exact_rerank(
                vectors,
                row_ids,
                proposed.row_ids,
                prototypes.vectors,
                selection.mask,
                settings.exact_edge_threshold,
                keep=search_count,
            )
            check_cancel()
            valid = reranked.row_ids[:, 0] >= 0
            neighbors[row_ids[valid]] = reranked.row_ids[valid, 0]
        check_cancel()
        neighbors.flush()
        return neighbors
    except BaseException:
        neighbors._mmap.close()
        raise
    finally:
        selection.close()


def _build_index_for_selection(
    prototypes: PrototypeStore,
    selection: PrototypeSelection,
    settings: object,
    check_cancel: Callable[[], None],
) -> SpeakerCandidateIndex | None:
    if selection.count == 0:
        return None
    check_cancel()
    index_settings = FaissIndexSettings(
        settings.prototype_index_factory,
        settings.training_rows,
        settings.prototype_search_effort,
        settings.random_seed,
    )
    index = SpeakerCandidateIndex(prototypes.dimension, index_settings)
    if index.requires_training:
        reservoir = DeterministicVectorReservoir(
            settings.training_rows, settings.random_seed
        )
        for row_ids, vectors in _prototype_blocks(
            prototypes, selection, settings.block_rows, check_cancel
        ):
            reservoir.add(row_ids, vectors)
        check_cancel()
        index.train(reservoir.result())
        check_cancel()
    for row_ids, vectors in _prototype_blocks(
        prototypes, selection, settings.block_rows, check_cancel
    ):
        check_cancel()
        index.add(row_ids, vectors)
    check_cancel()
    return index


def assignment_blocks(
    blocks: Iterable[EmbeddingBlock],
    labels: np.ndarray,
    prototypes: PrototypeStore,
    index: SpeakerCandidateIndex | None,
    established: PrototypeSelection,
    settings: object,
    check_cancel: Callable[[], None],
) -> Iterator[list[AssignmentRow]]:
    policy = AssignmentPolicy(
        accept_threshold=settings.accept_threshold,
        min_margin=settings.min_margin,
        new_threshold=settings.new_threshold,
        dispersion_penalty=settings.dispersion_penalty,
        threshold_version=settings.threshold_version,
    )
    for block in blocks:
        check_cancel()
        scores = _score_block(
            block, prototypes, index, established, settings, check_cancel
        )
        rows = []
        for position, row_id in enumerate(block.row_ids):
            quality = EmbeddingQuality(str(block.qualities[position]))
            rejection = block.rejection_reasons[position]
            provisional_cluster_id = int(labels[row_id])
            provisional_cluster_suspicious = provisional_cluster_id >= 0 and bool(
                prototypes.suspicious[provisional_cluster_id]
            )
            decision = decide(
                quality,
                None if rejection is None else str(rejection),
                provisional_cluster_id,
                provisional_cluster_suspicious,
                scores[position],
                policy,
            )
            rows.append(
                _assignment_row(block, position, int(row_id), quality, decision)
            )
        yield rows


def _prototype_blocks(
    prototypes: PrototypeStore,
    established: PrototypeSelection,
    block_rows: int,
    check_cancel: Callable[[], None],
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    for start in range(0, prototypes.item_count, block_rows):
        check_cancel()
        stop = min(start + block_rows, prototypes.item_count)
        row_ids = np.flatnonzero(established.mask[start:stop]) + start
        if len(row_ids):
            yield row_ids, np.asarray(prototypes.vectors[row_ids], dtype=np.float32)


def _score_block(
    block: EmbeddingBlock,
    prototypes: PrototypeStore,
    index: SpeakerCandidateIndex | None,
    established: PrototypeSelection,
    settings: object,
    check_cancel: Callable[[], None],
) -> list[CandidateScores | None]:
    result: list[CandidateScores | None] = [None] * len(block.row_ids)
    accepted_positions = np.flatnonzero(block.accepted_mask)
    if index is None or not len(accepted_positions):
        return result
    neighbors = min(settings.assignment_neighbors, index.item_count)
    check_cancel()
    proposed = index.search(block.embeddings[accepted_positions], neighbors)
    check_cancel()
    reranked = exact_rerank(
        block.embeddings[accepted_positions],
        block.row_ids[accepted_positions],
        proposed.row_ids,
        prototypes.vectors,
        established.mask,
        -1.0,
        keep=neighbors,
        exclude_self=False,
    )
    check_cancel()
    for output_position, block_position in enumerate(accepted_positions):
        valid = reranked.row_ids[output_position] >= 0
        cluster_ids = reranked.row_ids[output_position][valid].astype(int).tolist()
        candidate_scores = (
            reranked.scores[output_position][valid].astype(float).tolist()
        )
        if cluster_ids:
            result[block_position] = CandidateScores(
                cluster_ids=cluster_ids,
                scores=candidate_scores,
                best_dispersion=float(prototypes.dispersion[cluster_ids[0]]),
                best_suspicious=bool(prototypes.suspicious[cluster_ids[0]]),
            )
    return result


def _assignment_row(
    block: EmbeddingBlock,
    position: int,
    row_id: int,
    quality: EmbeddingQuality,
    decision: AssignmentDecision,
) -> AssignmentRow:
    true_label = block.true_labels[position]
    return AssignmentRow(
        row_id=row_id,
        segment_id=str(block.segment_ids[position]),
        audio_id=str(block.audio_ids[position]),
        duration_seconds=float(block.duration_seconds[position]),
        quality=quality,
        outcome=decision.outcome,
        cluster_id=decision.cluster_id,
        best_cluster_id=decision.best_cluster_id,
        second_cluster_id=decision.second_cluster_id,
        best_score=decision.best_score,
        second_score=decision.second_score,
        margin=decision.margin,
        candidate_cluster_ids=decision.candidate_cluster_ids,
        candidate_scores=decision.candidate_scores,
        threshold_version=decision.threshold_version,
        reason=decision.reason,
        true_label=None if true_label is None else str(true_label),
        rejection_reason=decision.rejection_reason,
    )

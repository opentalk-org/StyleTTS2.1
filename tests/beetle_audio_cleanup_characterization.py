import inspect
import json
import sys
from dataclasses import asdict
from pathlib import Path
from uuid import UUID

from shared.audio_annotations import AudioAnnotations
from shared.db.audio import crud as audio_crud
from shared.db.audio.segment_references_crud import SegmentReference
from runner.nodes.training.beetle.config.data import GroupSamplingConfig
from runner.nodes.training.beetle.data.index import DatabaseSegmentIndex
from runner.nodes.training.beetle.data.sampling import (
    ContinuousBatchPlanner,
    DistributedShard,
)

PUBLIC_NAMES = (
    "AudioPackConfig",
    "audio_bucket_locations",
    "audio_file_annotations",
    "bulk_apply_speaker_assignments",
    "bulk_create_audio_files",
    "bulk_delete_audio_files",
    "bulk_read_audio_files",
    "bulk_replace_audio_segments",
    "bulk_update_audio_files",
    "bulk_update_audio_scores",
    "count_audio_file_references",
    "count_segment_references",
    "create_audio_file",
    "delete_audio_file",
    "get_audio_file",
    "get_audio_files_bulk",
    "list_audio_file_references_page",
    "list_audio_files_by_run",
    "list_audio_segments_bulk",
    "list_segment_references_page",
    "prune_audio_packs",
    "read_audio_file",
    "read_audio_part",
    "search_audio_file_ids",
    "search_audio_files",
    "update_audio_file",
)


def reference(
    audio_number: int,
    segment_index: int,
    speaker_id: str,
) -> SegmentReference:
    audio_id = UUID(int=audio_number)
    start = float(segment_index * 3)
    end = start + 2.5
    segment_id = f"{audio_number}-{segment_index}"
    return SegmentReference(
        audio_file_id=audio_id,
        audio_name=f"audio-{audio_number}",
        audio_duration=12.0,
        annotations=AudioAnnotations(metadata={"sample_rate": 24000}),
        audio_byte_length=240000,
        audio_virtual=False,
        audio_storage_kind="packed",
        language="en",
        style_prompt=f"style-{audio_number}",
        voice_prompt=f"voice-{speaker_id}",
        segment_index=segment_index,
        segment={
            "id": segment_id,
            "start": start,
            "end": end,
            "text": "one two",
            "phon": "w ah n t uw",
            "alignment": [
                {"word": "one", "start": start, "end": start + 1.0},
                {"word": "two", "start": start + 1.2, "end": end},
            ],
            "annotations": {"speaker_id": speaker_id},
        },
    )


def assert_catalog_owners() -> None:
    from shared.db.audio import catalog, segment_catalog, segments

    assert audio_crud.get_audio_files_bulk is catalog.get_audio_files_bulk
    assert (
        audio_crud.list_audio_file_references_page
        is catalog.list_audio_file_references_page
    )
    assert (
        audio_crud.list_segment_references_page
        is segment_catalog.list_segment_references_page
    )
    assert audio_crud.list_audio_segments_bulk is segments.list_audio_segments_bulk


def main(output_path: Path) -> None:
    assert_catalog_owners()
    references = [
        reference(
            audio_number,
            segment_index,
            f"speaker-{(audio_number - 1) // 2}",
        )
        for audio_number in range(1, 5)
        for segment_index in range(2)
    ]
    index = DatabaseSegmentIndex.from_references(
        UUID(int=100),
        (),
        ("en",),
        8.0,
        references,
    )
    grouping = GroupSamplingConfig(
        voices_per_batch=2,
        utterances_per_voice=2,
        recordings_per_batch=2,
        cuts_per_recording=2,
    )
    planner = ContinuousBatchPlanner(
        index,
        batch_size=2,
        seed=17,
        maximum_seconds=8.0,
        grouping=grouping,
        shard=DistributedShard(0, 1),
    )
    first = planner.next_window(2)
    saved = planner.state_dict()
    expected_next = planner.next_window(2)
    restored = ContinuousBatchPlanner(
        index,
        batch_size=2,
        seed=17,
        maximum_seconds=8.0,
        grouping=grouping,
        shard=DistributedShard(0, 1),
    )
    restored.load_state_dict(saved)
    actual_next = restored.next_window(2)
    assert actual_next == expected_next
    payload = {
        "facade": {
            name: (
                str(inspect.signature(getattr(audio_crud, name)))
                if callable(getattr(audio_crud, name))
                else type(getattr(audio_crud, name)).__name__
            )
            for name in PUBLIC_NAMES
        },
        "fingerprint": index.fingerprint,
        "report": asdict(index.report),
        "segments": [asdict(key) for key in index.pools.segments],
        "validation": asdict(index.validation),
        "first": [asdict(item) for item in first],
        "saved": asdict(saved),
        "next": [asdict(item) for item in expected_next],
    }
    encoded = json.dumps(payload, sort_keys=True, default=str, indent=2)
    if output_path.exists():
        assert json.loads(output_path.read_text()) == json.loads(encoded)
    else:
        output_path.write_text(encoded)


if __name__ == "__main__":
    main(Path(sys.argv[1]))

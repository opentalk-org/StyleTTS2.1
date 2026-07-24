from sqlalchemy.orm import Session

from shared.db.audio.catalog import (
    audio_bucket_locations,
    audio_file_annotations,
    count_audio_file_references,
    get_audio_file,
    get_audio_files_bulk,
    list_audio_file_references_page,
    list_audio_files,
    list_audio_files_by_run,
    search_audio_file_ids,
    search_audio_files,
)
from shared.db.audio.files import (
    _object_store,
    bulk_create_audio_files,
    bulk_delete_audio_files,
    bulk_read_audio_files,
    bulk_read_audio_parts,
    bulk_update_audio_files,
    create_audio_file,
    delete_audio_file,
    read_audio_file,
    read_audio_part,
    update_audio_file,
)
from shared.db.audio.pack_cleanup import purge_orphaned_audio_packs as purge_orphaned_audio_packs
from shared.db.audio.pack_prune import prune_fragmented_audio_packs
from shared.db.audio.pack_store import AudioPackConfig, ObjectStore
from shared.db.audio.scores_crud import bulk_update_audio_scores
from shared.db.audio.speaker_assignment_crud import (
    bulk_apply_speaker_assignments as bulk_apply_speaker_assignments,
)
from shared.db.audio.segment_catalog import (
    count_segment_references,
    list_segment_references_page,
)
from shared.db.audio.segments import (
    bulk_replace_audio_segments,
    create_segment,
    delete_segment,
    list_audio_segments,
    list_audio_segments_bulk,
    replace_audio_segments,
    update_segment,
    update_segment_phonemes,
    update_segment_text,
)


def prune_audio_packs(
    session: Session,
    store: ObjectStore | None = None,
    config: AudioPackConfig = AudioPackConfig(),
) -> None:
    prune_fragmented_audio_packs(session, _object_store(session, store), config)

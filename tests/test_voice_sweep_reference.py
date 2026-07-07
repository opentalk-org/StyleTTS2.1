from __future__ import annotations

import unittest
from types import SimpleNamespace
from uuid import UUID, uuid4

from runner.nodes.testing.nodes import _voice_reference_audio_id


class VoiceSweepReferenceTests(unittest.TestCase):
    def test_uses_longest_audio_file_with_matching_metadata_voice_id(self) -> None:
        voice_id = uuid4()
        short_id = uuid4()
        long_id = uuid4()
        audio_files = [
            SimpleNamespace(id=short_id, duration=1.0, metadata_={"voice_id": str(voice_id)}, segments=[]),
            SimpleNamespace(id=long_id, duration=3.0, metadata_={"voice_id": str(voice_id)}, segments=[]),
        ]

        self.assertEqual(_voice_reference_audio_id(audio_files, voice_id), long_id)

    def test_uses_segment_voice_id_when_metadata_is_missing(self) -> None:
        voice_id = uuid4()
        audio_id = uuid4()
        audio_files = [
            SimpleNamespace(id=audio_id, duration=1.0, metadata_={}, segments=[{"voice_id": str(voice_id)}]),
        ]

        self.assertEqual(_voice_reference_audio_id(audio_files, voice_id), audio_id)

    def test_uses_metadata_speaker_when_voice_id_is_missing(self) -> None:
        voice_id = uuid4()
        audio_id = uuid4()
        audio_files = [
            SimpleNamespace(id=audio_id, duration=1.0, metadata_={"speaker": "spk_voice_15b5db"}, segments=[]),
        ]

        self.assertEqual(_voice_reference_audio_id(audio_files, voice_id, "spk_voice_15b5db"), audio_id)

    def test_uses_segment_speaker_when_voice_id_is_missing(self) -> None:
        voice_id = uuid4()
        audio_id = uuid4()
        audio_files = [
            SimpleNamespace(id=audio_id, duration=1.0, metadata_={}, segments=[{"speaker": "spk_voice_15b5db"}]),
        ]

        self.assertEqual(_voice_reference_audio_id(audio_files, voice_id, "spk_voice_15b5db"), audio_id)

    def test_raises_when_voice_has_no_audio_reference(self) -> None:
        with self.assertRaisesRegex(KeyError, "Voice has no audio reference"):
            _voice_reference_audio_id([], UUID("25cc94ca-fb1b-4e87-a666-ba48a6c8fbc8"))


if __name__ == "__main__":
    unittest.main()

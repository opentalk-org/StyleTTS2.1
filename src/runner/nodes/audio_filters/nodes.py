from runflow.core.node import Node
from runflow.core.settings import StrictSettings
from runflow.policies import BatchMode, BatchPolicy
from runflow.runtime.output_router import INPUT_INDEX_OUTPUT
from runner.nodes.datatypes import AudioPort
from runner.nodes.models import Audio


class FilterAudioLanguageSettings(StrictSettings):
    excluded_languages: str = ""


class FilterAudioLanguageNode(Node):
    NODE_TYPE = "FilterAudioLanguage"
    DESCRIPTION = "Drop audio records whose language is missing, blank, or included in the excluded languages setting. Language matching is case-insensitive and treats underscores as hyphens."
    CATEGORY = "Audio"
    SETTINGS = FilterAudioLanguageSettings
    INPUTS = {"audio": AudioPort()}
    OUTPUTS = {"audio": AudioPort()}
    BATCH_POLICY = BatchPolicy(BatchMode.MICRO_BATCH, preferred_size=256, max_size=256)

    async def execute(self, batch, context):
        excluded_languages = {
            _normalized_language(language)
            for language in self.settings.excluded_languages.split(",")
        }
        outputs = []
        for input_index, inputs in enumerate(batch):
            context.check_cancel()
            audio: Audio = inputs["audio"]
            language = _normalized_language(audio.language)
            if language and language not in excluded_languages:
                outputs.append({INPUT_INDEX_OUTPUT: input_index, "audio": audio})
        return outputs


class FilterProcessedAudioNode(Node):
    NODE_TYPE = "FilterProcessedAudio"
    DESCRIPTION = "Pass audio records that contain at least one text-bearing segment without phonemes, and drop records with no phonemization work remaining."
    CATEGORY = "Audio"
    INPUTS = {"audio": AudioPort()}
    OUTPUTS = {"audio": AudioPort()}
    BATCH_POLICY = BatchPolicy(BatchMode.MICRO_BATCH, preferred_size=256, max_size=256)

    async def execute(self, batch, context):
        outputs = []
        for input_index, inputs in enumerate(batch):
            context.check_cancel()
            audio: Audio = inputs["audio"]
            has_unprocessed_segment = any(
                segment.text.strip() and not segment.phon.strip()
                for segment in audio.segments
            )
            if has_unprocessed_segment:
                outputs.append({INPUT_INDEX_OUTPUT: input_index, "audio": audio})
        return outputs


def _normalized_language(language: str | None) -> str:
    return "" if language is None else language.strip().lower().replace("_", "-")

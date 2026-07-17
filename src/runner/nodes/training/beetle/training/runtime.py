from dataclasses import dataclass
from pathlib import Path

from transformers import BertModel, BertTokenizerFast


@dataclass(frozen=True)
class PhonemeResources:
    model: BertModel
    tokenizer: BertTokenizerFast


def load_phoneme_resources(model_path: Path) -> PhonemeResources:
    model = BertModel.from_pretrained(model_path, local_files_only=True)
    tokenizer = BertTokenizerFast.from_pretrained(
        model_path,
        local_files_only=True,
    )
    return PhonemeResources(model, tokenizer)

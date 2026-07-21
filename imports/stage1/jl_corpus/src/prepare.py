from pathlib import Path

from imports.stage1.common.transcribed_parquet import DatasetConfig, prepare_dataset


CONFIG = DatasetConfig("JL Corpus", "jl_corpus", "jlcorpus-*.parquet", 1, 2_400, "https://huggingface.co/datasets/HelloBug1/EMO-MEAD-Transcribed", None)


if __name__ == "__main__":
    prepare_dataset(Path(__file__).resolve().parent.parent, CONFIG)

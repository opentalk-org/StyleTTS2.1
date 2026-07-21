from pathlib import Path

from imports.stage1.common.transcribed_parquet import DatasetConfig, prepare_dataset


CONFIG = DatasetConfig("CREMA-D", "crema_d", "cremad-*.parquet", 2, 7_442, "https://huggingface.co/datasets/HelloBug1/EMO-MEAD-Transcribed", "crema_d")


if __name__ == "__main__":
    prepare_dataset(Path(__file__).resolve().parent.parent, CONFIG)

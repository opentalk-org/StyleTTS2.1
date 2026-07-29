import argparse
from pathlib import Path

from stage1_backend_verify import verify_stage_paths


STAGE_ROOT = Path(__file__).resolve().parent / "stage1"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fully verify and prune one uploaded Stage 1 dataset"
    )
    parser.add_argument("slug")
    return parser.parse_args()


def main() -> None:
    slug = arguments().slug
    verify_stage_paths([STAGE_ROOT / slug / "data.json"], prune_verified=True)


if __name__ == "__main__":
    main()

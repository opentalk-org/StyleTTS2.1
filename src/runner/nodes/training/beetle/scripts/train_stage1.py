from collections.abc import Sequence

from ..training.state import StageKind
from .common import run_cli


def main(argv: Sequence[str] | None = None) -> None:
    run_cli(StageKind.STAGE1, argv)


if __name__ == "__main__":
    main()

"""Prefetch Uni-pKa model checkpoints."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from .models import AVAILABLE_FOLDS, DEFAULT_FOLD, prefetch_checkpoints
from .paths import default_model_dir


def download_unipka_model(
    output_folder: str | Path | None = None,
    *,
    folds: Sequence[int] = (DEFAULT_FOLD,),
) -> tuple[Path, ...]:
    """Download and verify selected Uni-pKa folds."""

    return prefetch_checkpoints(
        folds,
        model_dir=output_folder or default_model_dir(),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download selected Uni-pKa model checkpoints. By default, fold 1 "
            "is stored in unipkainfer's per-user data directory."
        )
    )
    parser.add_argument(
        "--output-folder",
        type=Path,
        default=None,
        help=(
            "Destination for fold_N/checkpoint_best.pt files "
            f"(default: {default_model_dir()})."
        ),
    )
    parser.add_argument(
        "--fold",
        dest="folds",
        type=int,
        nargs="+",
        choices=AVAILABLE_FOLDS,
        default=(DEFAULT_FOLD,),
        metavar="N",
        help=(
            "Fold(s) to prefetch (default: 1), for example: "
            "--fold 1 or --fold 0 1 2 3 4."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        checkpoints = download_unipka_model(
            args.output_folder,
            folds=args.folds,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        _parser().exit(1, f"error: {exc}\n")

    destination = (args.output_folder or default_model_dir()).expanduser().resolve()
    print(f"Prepared {len(checkpoints)} Uni-pKa checkpoint(s) in {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Download and cache Uni-pKa model checkpoints."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pooch

from .paths import default_model_dir


MODEL_REPOSITORY = "bindresearch/pkasso_unipka"
MODEL_REVISION = "e28a5917f80556f026acee909fd6b6989d915e54"
DEFAULT_FOLD = 1
MODEL_REGISTRY = {
    "fold_0/checkpoint_best.pt": "sha256:4d5a5fc955f29f456518a58e87de5e3c86737660c4cd512aa058cd049062a200",
    "fold_1/checkpoint_best.pt": "sha256:6bf5e354860c717beb8d5eb3968a9ddbd526fc8e3defebdcad12074723f8b3b1",
    "fold_2/checkpoint_best.pt": "sha256:718ae5153ce98fead33435c4afbf99ffda2f2a2fd993c2eb2c5ef0dd81345b58",
    "fold_3/checkpoint_best.pt": "sha256:4b5b9cbf36f899c1d5cc0c9d85cf71a538a0d118d95b3f2e6eb98bd7bcc9d734",
    "fold_4/checkpoint_best.pt": "sha256:76f280d7a9d16097ced43e585b31d0766dffafc49d8cdc63f191b04890388f23",
}
AVAILABLE_FOLDS = tuple(range(len(MODEL_REGISTRY)))


def _checkpoint_name(fold: int) -> str:
    if fold not in AVAILABLE_FOLDS:
        available = ", ".join(str(item) for item in AVAILABLE_FOLDS)
        raise ValueError(f"Unknown Uni-pKa fold {fold}. Available folds: {available}.")
    return f"fold_{fold}/checkpoint_best.pt"


def _model_manager(model_dir: str | Path) -> pooch.Pooch:
    return pooch.create(
        path=Path(model_dir).expanduser().resolve(),
        base_url=(
            f"https://huggingface.co/{MODEL_REPOSITORY}/resolve/"
            f"{MODEL_REVISION}/"
        ),
        registry=MODEL_REGISTRY,
    )


def ensure_checkpoint(
    model_dir: str | Path,
    fold: int,
    *,
    progressbar: bool = True,
) -> Path:
    """Return a verified local checkpoint, downloading it when necessary."""

    checkpoint = _model_manager(model_dir).fetch(
        _checkpoint_name(fold),
        progressbar=progressbar,
    )
    return Path(checkpoint)


def prefetch_checkpoints(
    folds: Iterable[int] = (DEFAULT_FOLD,),
    *,
    model_dir: str | Path | None = None,
    progressbar: bool = True,
) -> tuple[Path, ...]:
    """Download and verify the selected model folds."""

    destination = model_dir or default_model_dir()
    return tuple(
        ensure_checkpoint(destination, fold, progressbar=progressbar)
        for fold in folds
    )

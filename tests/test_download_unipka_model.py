from pathlib import Path

import pytest

import download_model as downloader
from unipkainfer.pka_predictor.api import _missing_checkpoint_message


def test_downloads_fold_checkpoints_to_required_output_folder(
    monkeypatch,
    tmp_path,
):
    captured = {}
    output_folder = tmp_path / "models"

    def mock_snapshot_download(**kwargs):
        captured.update(kwargs)
        checkpoint = Path(kwargs["local_dir"]) / "fold_3" / "checkpoint_best.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"checkpoint")
        return str(kwargs["local_dir"])

    monkeypatch.setattr(downloader, "_snapshot_download", mock_snapshot_download)

    result = downloader.download_unipka_model(
        output_folder,
        revision="release-1",
        force_download=True,
    )

    assert result == (output_folder / "fold_3" / "checkpoint_best.pt",)
    assert captured == {
        "repo_id": "bindresearch/pkasso_unipka",
        "revision": "release-1",
        "local_dir": output_folder,
        "allow_patterns": ["fold_*/checkpoint_best.pt"],
        "force_download": True,
    }


def test_uses_default_user_data_model_folder(monkeypatch, tmp_path):
    default_folder = tmp_path / "user-data" / "models"
    captured = {}

    monkeypatch.setattr(downloader, "default_model_dir", lambda: default_folder)

    def mock_snapshot_download(**kwargs):
        captured.update(kwargs)
        checkpoint = Path(kwargs["local_dir"]) / "fold_0" / "checkpoint_best.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.touch()
        return str(kwargs["local_dir"])

    monkeypatch.setattr(downloader, "_snapshot_download", mock_snapshot_download)

    result = downloader.download_unipka_model()

    assert result == (default_folder / "fold_0" / "checkpoint_best.pt",)
    assert captured["local_dir"] == default_folder


def test_fails_if_repository_contains_no_expected_checkpoints(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        downloader,
        "_snapshot_download",
        lambda **kwargs: str(kwargs["local_dir"]),
    )

    with pytest.raises(RuntimeError, match="No Uni-pKa checkpoints"):
        downloader.download_unipka_model(tmp_path)


def test_missing_checkpoint_message_recommends_downloader_for_model_folder():
    message = _missing_checkpoint_message(
        Path("/models/with spaces/fold_3/checkpoint_best.pt"),
        Path("/models/with spaces"),
    )

    assert (
        "unipka-download-model --output-folder '/models/with spaces'"
        in message
    )


def test_missing_default_checkpoint_message_uses_short_command(monkeypatch, tmp_path):
    model_dir = tmp_path / "user-data" / "models"
    monkeypatch.setattr(
        "unipkainfer.pka_predictor.api.default_model_dir",
        lambda: model_dir,
    )

    message = _missing_checkpoint_message(
        model_dir / "fold_0" / "checkpoint_best.pt",
        model_dir,
    )

    assert "`unipka-download-model`" in message
    assert "--output-folder" not in message

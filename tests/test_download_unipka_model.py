import pytest

from unipkainfer import models
from unipkainfer import download_model as downloader


def test_model_manager_preserves_fold_subdirectories(tmp_path):
    manager = models._model_manager(tmp_path / "models")

    assert manager.path == (tmp_path / "models").resolve()
    assert manager.base_url.endswith(f"/resolve/{models.MODEL_REVISION}/")
    assert set(manager.registry) == {
        f"fold_{fold}/checkpoint_best.pt" for fold in models.AVAILABLE_FOLDS
    }


def test_ensure_checkpoint_fetches_only_requested_fold(monkeypatch, tmp_path):
    captured = {}
    checkpoint = tmp_path / "models" / "fold_1" / "checkpoint_best.pt"

    class FakeManager:
        def fetch(self, name, *, progressbar):
            captured.update(name=name, progressbar=progressbar)
            return str(checkpoint)

    monkeypatch.setattr(models, "_model_manager", lambda model_dir: FakeManager())

    result = models.ensure_checkpoint(tmp_path / "models", 1)

    assert result == checkpoint
    assert captured == {
        "name": "fold_1/checkpoint_best.pt",
        "progressbar": True,
    }


def test_ensure_checkpoint_rejects_unknown_fold(tmp_path):
    with pytest.raises(ValueError, match="Unknown Uni-pKa fold 5"):
        models.ensure_checkpoint(tmp_path, 5)


def test_prefetch_command_defaults_to_fold_one(monkeypatch, tmp_path):
    captured = {}
    checkpoint = tmp_path / "models" / "fold_1" / "checkpoint_best.pt"

    def mock_prefetch(folds, *, model_dir, progressbar=True):
        captured.update(folds=tuple(folds), model_dir=model_dir)
        return (checkpoint,)

    monkeypatch.setattr(downloader, "prefetch_checkpoints", mock_prefetch)

    result = downloader.download_unipka_model(tmp_path / "models")

    assert result == (checkpoint,)
    assert captured == {
        "folds": (1,),
        "model_dir": tmp_path / "models",
    }


def test_prefetch_command_accepts_selected_folds(monkeypatch, tmp_path):
    captured = {}

    def mock_download(output_folder, *, folds):
        captured.update(output_folder=output_folder, folds=tuple(folds))
        return tuple(
            tmp_path / "models" / f"fold_{fold}" / "checkpoint_best.pt"
            for fold in folds
        )

    monkeypatch.setattr(downloader, "download_unipka_model", mock_download)

    assert downloader.main(
        ["--output-folder", str(tmp_path / "models"), "--fold", "1", "3"]
    ) == 0
    assert captured == {
        "output_folder": tmp_path / "models",
        "folds": (1, 3),
    }

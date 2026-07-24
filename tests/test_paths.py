from pathlib import Path

from unipkainfer import paths


def test_default_model_dir_uses_platform_user_data_directory(monkeypatch):
    monkeypatch.setattr(
        paths,
        "user_data_path",
        lambda *args, **kwargs: Path("/user/data/unipkainfer"),
    )

    assert paths.default_model_dir() == Path("/user/data/unipkainfer/models")

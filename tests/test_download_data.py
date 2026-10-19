import pytest

from src import download_data
from tests.helpers import write_raw_file


def test_existing_file_is_not_downloaded_again(tmp_path, monkeypatch):
    dest = write_raw_file(tmp_path / "raw.data", n=303)

    def boom(_):
        raise AssertionError("network must not be used when the file exists")

    monkeypatch.setattr(download_data, "_download_from_url", boom)
    assert download_data.download(dest=dest) == dest


def test_falls_back_to_ucimlrepo_when_direct_download_fails(tmp_path, monkeypatch):
    dest = tmp_path / "raw" / "heart.data"

    def fail(_):
        raise OSError("blocked")

    def fallback(path):
        write_raw_file(path, n=303)

    monkeypatch.setattr(download_data, "_download_from_url", fail)
    monkeypatch.setattr(download_data, "_download_with_ucimlrepo", fallback)
    assert download_data.download(dest=dest).exists()


def test_clear_error_when_every_source_fails(tmp_path, monkeypatch):
    def fail(_):
        raise OSError("blocked")

    monkeypatch.setattr(download_data, "_download_from_url", fail)
    monkeypatch.setattr(download_data, "_download_with_ucimlrepo", fail)
    with pytest.raises(RuntimeError, match="download"):
        download_data.download(dest=tmp_path / "x.data")
    assert not (tmp_path / "x.data").exists()


def test_verify_rejects_wrong_column_count(tmp_path):
    bad = tmp_path / "bad.data"
    bad.write_text("1,2,3\n4,5,6\n")
    with pytest.raises(ValueError, match="columns"):
        download_data._verify(bad)

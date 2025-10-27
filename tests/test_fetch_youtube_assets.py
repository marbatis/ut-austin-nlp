from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import fetch_youtube_assets as fetch


def test_load_index_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"

    result = fetch.load_index(missing)

    assert result == {"weeks": []}


def test_load_index_rejects_invalid_json(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("not json", encoding="utf-8")

    with pytest.raises(ValueError):
        fetch.load_index(bad)


def test_main_returns_early_without_index(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    output_dir = tmp_path / "output"

    fetch.main(index_path=tmp_path / "missing.json", out_dir=output_dir)

    captured = capsys.readouterr()
    assert "No course index" in captured.out
    assert not output_dir.exists()


def test_main_no_videos(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    index_path = tmp_path / "index.json"
    index_path.write_text(
        json.dumps({"weeks": [{"items": [{"type": "reading", "url": "https://example.com"}]}]}),
        encoding="utf-8",
    )

    calls = {"info": 0, "transcript": 0}

    def fake_info(url: str) -> dict:
        calls["info"] += 1
        return {}

    def fake_transcript(video_id: str) -> None:
        calls["transcript"] += 1
        return None

    monkeypatch.setattr(fetch, "get_info", fake_info)
    monkeypatch.setattr(fetch, "get_transcript", fake_transcript)

    out_dir = tmp_path / "data"
    fetch.main(index_path=index_path, out_dir=out_dir)

    assert calls == {"info": 0, "transcript": 0}
    video_info = json.loads((out_dir / "video_info.json").read_text(encoding="utf-8"))
    assert video_info == {}

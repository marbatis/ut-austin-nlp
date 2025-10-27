from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

try:  # pragma: no cover - optional dependency
    from youtube_transcript_api import (
        NoTranscriptFound,
        TranscriptsDisabled,
        YouTubeTranscriptApi,
    )
except ImportError:  # pragma: no cover - optional dependency
    NoTranscriptFound = TranscriptsDisabled = YouTubeTranscriptApi = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    from yt_dlp import YoutubeDL
except ImportError:  # pragma: no cover - optional dependency
    YoutubeDL = None  # type: ignore[assignment]


def vid_from_url(url: str) -> str:
    if "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0]
    if "youtube.com/watch" in url:
        return parse_qs(urlparse(url).query).get("v", [""])[0]
    return ""


def get_info(url: str) -> dict:
    if YoutubeDL is None:  # pragma: no cover - defensive
        raise RuntimeError("yt-dlp is required to fetch video metadata.")

    ydl_opts = {"quiet": True, "skip_download": True}
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
    return {
        "id": info.get("id"),
        "title": info.get("title"),
        "channel": info.get("channel"),
        "uploader": info.get("uploader"),
        "upload_date": info.get("upload_date"),
        "duration": info.get("duration"),
        "webpage_url": info.get("webpage_url"),
        "description": (info.get("description", "") or "")[:1000],
    }


def get_transcript(video_id: str):
    if YouTubeTranscriptApi is None:  # pragma: no cover - defensive
        raise RuntimeError("youtube-transcript-api is required to fetch transcripts.")

    languages = ["en", "en-US", "en-GB"]
    try:
        return YouTubeTranscriptApi.get_transcript(video_id, languages=languages)
    except (TranscriptsDisabled, NoTranscriptFound):
        try:
            listing = YouTubeTranscriptApi.list_transcripts(video_id)
            transcript = listing.find_generated_transcript(
                listing._translation_languages or ["en"]
            )
            return transcript.fetch()
        except Exception:
            return None


def load_index(path: str | Path = "course_index.json") -> dict[str, Any]:
    file_path = Path(path)

    if not file_path.exists():
        return {"weeks": []}

    raw_text = file_path.read_text(encoding="utf-8")
    if not raw_text.strip():
        return {"weeks": []}

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Invalid JSON in {file_path}") from exc

    if not isinstance(data, dict):  # pragma: no cover - defensive
        raise ValueError(f"Expected object at top level in {file_path}")

    data.setdefault("weeks", [])
    return data


def main(index_path: str = "course_index.json", out_dir: str = "data") -> None:
    index = load_index(index_path)

    if not index.get("weeks"):
        print(f"No course index found at {index_path}; skipping download.")
        return

    base = Path(out_dir)
    transcripts_dir = base / "transcripts"
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    video_info = {}

    for week in index.get("weeks", []):
        for item in week.get("items", []):
            if item.get("type") != "video":
                continue
            video_id = vid_from_url(item.get("url", ""))
            if not video_id:
                continue
            url = item["url"]
            try:
                info = get_info(url)
                video_info[video_id] = info
            except Exception as exc:  # noqa: BLE001
                print("info fail:", url, exc)
            try:
                transcript = get_transcript(video_id)
                if transcript:
                    json_path = transcripts_dir / f"{video_id}.json"
                    json_path.write_text(json.dumps(transcript, indent=2), encoding="utf-8")

                    words = []
                    for segment in transcript:
                        words.extend(segment["text"].split())
                        if len(words) >= 120:
                            break
                    excerpt = " ".join(words[:120])
                    (transcripts_dir / f"{video_id}.md").write_text(
                        (
                            "> Transcript excerpt (≈120 words)\n>\n> "
                            f"{excerpt}\n\n(Full transcript JSON in this folder.)"
                        ),
                        encoding="utf-8",
                    )
                    time.sleep(0.3)
            except Exception as exc:  # noqa: BLE001
                print("transcript fail:", video_id, exc)

    (base / "video_info.json").write_text(
        json.dumps(video_info, indent=2), encoding="utf-8"
    )
    print("Saved video_info.json and transcripts under", base)


if __name__ == "__main__":
    if len(sys.argv) > 2:
        main(sys.argv[1], sys.argv[2])
    elif len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main()

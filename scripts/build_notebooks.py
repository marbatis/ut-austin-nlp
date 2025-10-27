import json
import re
from pathlib import Path

import nbformat as nbf


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "item"


def markdown_cell(content: str):
    return nbf.v4.new_markdown_cell(content)


def load_json(path: str):
    file_path = Path(path)
    if not file_path.exists():
        return {}
    return json.loads(file_path.read_text(encoding="utf-8"))


def video_id_from_url(url: str) -> str:
    from urllib.parse import parse_qs, urlparse

    if "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0]
    if "youtube.com/watch" in url:
        return parse_qs(urlparse(url).query).get("v", [""])[0]
    return ""


def build_notebook(
    week_dir: Path, item: dict, readings: list, video_info: dict, transcripts_dir: Path
) -> Path:
    notebook = nbf.v4.new_notebook()
    title = item["title"]
    url = item["url"]
    video_id = video_id_from_url(url)
    info = video_info.get(video_id, {})

    notebook.cells.extend(
        [
            markdown_cell(f"# {title}\n\n- 📺 **Video:** [{url}]({url})"),
            markdown_cell("## Overview\n- What you’ll learn\n- Why it matters\n"),
            markdown_cell("## Key ideas\n- (Fill in after watching)"),
        ]
    )

    if info:
        details = []
        if info.get("channel"):
            details.append(f"- **Channel:** {info['channel']}")
        if info.get("upload_date"):
            details.append(f"- **Uploaded:** {info['upload_date']}")
        if info.get("duration"):
            details.append(f"- **Duration:** {info['duration']}s")
        if details:
            notebook.cells.append(markdown_cell("## Video info\n" + "\n".join(details)))

    excerpt_path = transcripts_dir / f"{video_id}.md"
    if video_id and excerpt_path.exists():
        excerpt = excerpt_path.read_text(encoding="utf-8")
        notebook.cells.append(
            markdown_cell(
                "## Transcript (short excerpt)\n"
                f"{excerpt}\n\n> See `data/transcripts/{video_id}.json` for the full time-stamped transcript (if saved)."
            )
        )

    references = [
        f"- [{ref['title']}]({ref['url']})"
        for ref in readings
        if ref.get("type") == "reading"
    ]
    if references:
        notebook.cells.append(
            markdown_cell("## References\n" + "\n".join(references) + "\n")
        )

    notebook.cells.append(
        markdown_cell("*Links only; we do not redistribute slides/papers.*")
    )

    week_dir.mkdir(parents=True, exist_ok=True)
    output_path = week_dir / f"{slug(title)}.ipynb"
    nbf.write(notebook, output_path)
    return output_path


def main(
    index_path: str = "course_index.json",
    notebooks_dir: str = "notebooks",
    info_path: str = "data/video_info.json",
    transcripts_path: str = "data/transcripts",
) -> None:
    index = load_json(index_path)
    video_info = load_json(info_path)
    transcripts_dir = Path(transcripts_path)
    notebooks_root = Path(notebooks_dir)

    for week in index.get("weeks", []):
        match = re.match(r"Week\s+(\d+):\s*(.*)", week.get("week_title", ""))
        if match:
            folder_name = f"{int(match.group(1)):02d}-{slug(match.group(2))}"
        else:
            folder_name = slug(week.get("week_title", "Week"))
        week_dir = notebooks_root / folder_name

        for item in week.get("items", []):
            if item.get("type") == "video":
                build_notebook(week_dir, item, week.get("items", []), video_info, transcripts_dir)

    print("Notebooks under:", notebooks_root)


if __name__ == "__main__":
    main()

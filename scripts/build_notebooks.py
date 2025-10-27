"""Generate topic notebooks from the scraped course index."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import nbformat as nbf

if __package__ is None:  # pragma: no cover - runtime convenience
    sys.path.append(str(Path(__file__).resolve().parent))
    from demo_blocks import pick_blocks_for  # type: ignore
    from utils import ensure_directory, slugify  # type: ignore
else:  # pragma: no cover
    from .demo_blocks import pick_blocks_for
    from .utils import ensure_directory, slugify


def _markdown_cell(text: str) -> nbf.NotebookNode:
    cell = nbf.v4.new_markdown_cell(text)
    cell.metadata.setdefault("tags", [])
    return cell


def _code_cell(code: str, tags: Iterable[str] | None = None) -> nbf.NotebookNode:
    cell = nbf.v4.new_code_cell(code)
    if tags:
        cell.metadata["tags"] = list(tags)
    return cell


def _build_overview_cell(title: str, video_url: str, readings: List[Dict[str, str]]) -> str:
    lines = [f"# {title}", "", f"📺 **Watch:** [{video_url}]({video_url})"]
    if readings:
        lines.append("")
        lines.append("📄 **Readings:**")
        for reading in readings:
            lines.append(f"- [{reading['title']}]({reading['url']})")
    return "\n".join(lines)


def _setup_code() -> str:
    return (
        "import os\n"
        "import random\n"
        "import numpy as np\n"
        "import torch\n"
        "\n"
        "random.seed(0)\n"
        "np.random.seed(0)\n"
        "torch.manual_seed(0)\n"
        "CI = os.environ.get('CI', '').lower() == 'true'\n"
    )


def build_notebook(week_dir: Path, item: Dict[str, str], readings: List[Dict[str, str]]) -> Path:
    nb = nbf.v4.new_notebook()

    title = item["title"].strip()
    video_url = item.get("url", "")

    nb.cells.append(_markdown_cell(_build_overview_cell(title, video_url, readings)))
    nb.cells.append(_markdown_cell("## Overview\n\n- What you will learn\n- Why this topic matters"))
    nb.cells.append(_code_cell(_setup_code(), tags=["setup", "light"]))
    nb.cells.append(
        _markdown_cell(
            "## Key ideas\n\n- Summarise the central concepts after reviewing the lecture."
        )
    )

    demos = list(pick_blocks_for(title))
    if demos:
        for demo in demos:
            nb.cells.append(_markdown_cell(f"## Demo · {demo.title}"))
            nb.cells.append(_code_cell(demo.code, tags=demo.tags))
    else:
        nb.cells.append(_markdown_cell("## Demo\n\n- TODO: add a runnable example."))

    nb.cells.append(
        _markdown_cell(
            "## Try it yourself\n\n- Modify the example above and observe the effect.\n"
            "- Evaluate the model on new inputs."
        )
    )

    if readings:
        ref_lines = ["## Further reading"] + [
            f"- [{reading['title']}]({reading['url']})" for reading in readings
        ]
        nb.cells.append(_markdown_cell("\n".join(ref_lines)))

    nb.cells.append(
        _markdown_cell(
            "## Attribution\n\nBased on the UT Austin CS388/AI388/DSC395T online course materials."
        )
    )

    ensure_directory(week_dir)
    slug = slugify(title)
    output = week_dir / f"{slug}.ipynb"
    nbf.write(nb, output)
    return output


def load_index(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


WEEK_FOLDER_MAP = {
    "1": "01-intro-linear-classification",
    "2": "02-multiclass-and-neural",
    "3": "03-word-embeddings",
    "4": "04-language-modeling-and-self-attention",
    "5": "05-transformers-and-decoding",
    "6": "06-pretraining-and-seq2seq",
    "7": "07-08-structured-prediction",
    "8": "07-08-structured-prediction",
    "9": "09-modern-llms",
    "10": "10-explanations",
    "11": "11-qa-and-dialogue",
    "12": "12-mt-and-summarization",
    "13": "13-14-multilingual-grounding-ethics",
    "14": "13-14-multilingual-grounding-ethics",
}


def infer_week_folder(week_id: str, week_title: str) -> str:
    if week_id in WEEK_FOLDER_MAP:
        return WEEK_FOLDER_MAP[week_id]
    for key, folder in WEEK_FOLDER_MAP.items():
        if week_title.lower().startswith(f"week {key}"):
            return folder
    return slugify(week_title)


def generate_notebooks(index_path: Path, out_dir: Path) -> List[Path]:
    data = load_index(index_path)
    notebooks: List[Path] = []
    for week in data.get("weeks", []):
        folder_name = infer_week_folder(str(week.get("week_id", "")), week.get("week_title", ""))
        week_dir = out_dir / folder_name
        readings = [item for item in week.get("items", []) if item.get("type") != "video"]
        for item in week.get("items", []):
            if item.get("type") != "video":
                continue
            notebooks.append(build_notebook(week_dir, item, readings))
    return notebooks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True, type=Path, help="Path to course_index.json")
    parser.add_argument("--out", default=Path("notebooks"), type=Path, help="Notebook output directory")
    args = parser.parse_args()

    notebooks = generate_notebooks(args.index, args.out)
    print(f"Generated {len(notebooks)} notebooks in {args.out}")


if __name__ == "__main__":
    main()

import json
import re
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import nbformat as nbf


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "item"


def markdown_cell(content: str):
    return nbf.v4.new_markdown_cell(content)


def code_cell(content: str):
    return nbf.v4.new_code_cell(content)


def load_json(path: str):
    file_path = Path(path)
    if not file_path.exists():
        return {}
    return json.loads(file_path.read_text(encoding="utf-8"))


def video_id_from_url(url: str) -> str:
    if "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0]
    if "youtube.com/watch" in url:
        return parse_qs(urlparse(url).query).get("v", [""])[0]
    return ""


DEMO_LOGREG = """\
# Tiny logistic regression demo on synthetic data
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

X, y = make_classification(n_samples=800, n_features=10, random_state=0)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0)
clf = LogisticRegression(max_iter=500).fit(Xtr, ytr)
print("Accuracy:", accuracy_score(yte, clf.predict(Xte)))
"""

DEMO_ATTENTION = """\
# Scaled dot-product attention (toy)
import numpy as np

def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

np.random.seed(0)
Q = np.random.randn(3, 4)  # 3 queries, dim 4
K = np.random.randn(5, 4)  # 5 keys, dim 4
V = np.random.randn(5, 6)  # 5 values, dim 6

scores = Q @ K.T / np.sqrt(Q.shape[-1])  # (3,5)
weights = softmax(scores, axis=-1)       # (3,5)
out = weights @ V                        # (3,6)
print("weights.shape:", weights.shape, "out.shape:", out.shape)
"""


def pick_demo(title: str) -> str:
    lower = title.lower()
    if any(keyword in lower for keyword in ["perceptron", "logistic", "linear binary", "multiclass"]):
        return DEMO_LOGREG
    if "attention" in lower or "self-attention" in lower or "transformer" in lower:
        return DEMO_ATTENTION
    return "print('Try the exercises below and follow the linked materials.')"


def build_notebook(
    week_dir: Path, item: dict, readings: list, video_info: dict, transcripts_dir: Path
) -> Path:
    notebook = nbf.v4.new_notebook()
    title = item.get("title", "Lecture")
    url = item.get("url", "")
    video_id = video_id_from_url(url)
    info = video_info.get(video_id, {})

    notebook.cells.extend(
        [
            markdown_cell(f"# {title}\n\n- 📺 **Video:** [{url}]({url})"),
            markdown_cell("## Overview\n- What you’ll learn (fill in after watching)\n- Why it matters\n"),
            code_cell("import os, random\nrandom.seed(0)\nCI = os.environ.get('CI') == 'true'"),
            markdown_cell("## Key ideas\n- TODO: Summarize the core ideas after viewing the lecture."),
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
        excerpt = excerpt_path.read_text(encoding="utf-8").strip()
        if excerpt:
            notebook.cells.append(
                markdown_cell(
                    "## Transcript (excerpt)\n"
                    f"{excerpt}\n\n> Full transcript (if saved): `data/transcripts/{video_id}.json`."
                )
            )

    notebook.cells.append(markdown_cell("## Demo"))
    notebook.cells.append(code_cell(pick_demo(title)))
    notebook.cells.append(
        markdown_cell("## Try it\n- Modify the demo\n- Add a tiny dataset or counter-example\n")
    )

    reference_links = [
        f"- [{ref['title']}]({ref['url']})" for ref in readings if ref.get("type") == "reading"
    ]
    if reference_links:
        notebook.cells.append(
            markdown_cell("## References\n" + "\n".join(reference_links) + "\n")
        )

    notebook.cells.append(markdown_cell("*Links only; we do not redistribute slides or papers.*"))

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
        readings = [item for item in week.get("items", []) if item.get("type") == "reading"]

        for item in week.get("items", []):
            if item.get("type") == "video":
                build_notebook(week_dir, item, readings, video_info, transcripts_dir)

    print("Notebooks written under:", notebooks_root)


if __name__ == "__main__":
    main()

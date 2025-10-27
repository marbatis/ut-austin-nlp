import json, os, re, nbformat as nbf
from pathlib import Path

def slugify(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s or "item"

def md(text): 
    return nbf.v4.new_markdown_cell(text)

def code(text):
    return nbf.v4.new_code_cell(text)

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
out = weights @ V                         # (3,6)
print("weights.shape:", weights.shape, "out.shape:", out.shape)
"""

def pick_demo(title: str) -> str:
    t = title.lower()
    if any(k in t for k in ["perceptron", "logistic", "linear binary", "multiclass"]):
        return DEMO_LOGREG
    if "attention" in t or "self-attention" in t or "transformer" in t:
        return DEMO_ATTENTION
    return "print('Try the exercises below and follow the linked materials.')"  # default

def make_nb(week_dir: Path, item: dict, all_readings: list):
    nb = nbf.v4.new_notebook()
    title, url = item["title"], item["url"]

    # Title + links
    nb.cells.append(md(f"# {title}\n\n- Source: [{url}]({url})"))
    # Overview
    nb.cells.append(md("## Overview\n- What you’ll learn (fill in after watching)\n- Why it matters\n"))
    # Setup (seed / CI guard)
    nb.cells.append(code("import os, random\nrandom.seed(0)\nCI = os.environ.get('CI') == 'true'"))
    # Key ideas
    nb.cells.append(md("## Key ideas\n- TODO: Summarize the core ideas after viewing the lecture."))
    # Demo
    nb.cells.append(md("## Demo"))
    nb.cells.append(code(pick_demo(title)))
    # Exercises
    nb.cells.append(md("## Try it\n- Modify the demo\n- Add a tiny dataset / example\n"))
    # References
    if all_readings:
        refs = "\n".join(f"- [{r['title']}]({r['url']})" for r in all_readings)
        nb.cells.append(md(f"## References\n{refs}\n\n*Links only; we do not redistribute PDFs or slides.*"))

    week_dir.mkdir(parents=True, exist_ok=True)
    fname = week_dir / f"{slugify(title)}.ipynb"
    nbf.write(nb, fname)
    return fname

def main(index_path="course_index.json", out_dir="notebooks"):
    idx = json.loads(Path(index_path).read_text(encoding="utf-8"))
    base = Path(out_dir)

    for wk in idx["weeks"]:
        # "Week 1: Intro and Linear Classification" -> "01-intro-and-linear-classification"
        m = re.match(r"Week\s+(\d+):\s*(.*)", wk["week_title"])
        if m:
            num = int(m.group(1))
            name = slugify(m.group(2)) or f"week-{num}"
            week_folder = base / f"{num:02d}-{name}"
        else:
            week_folder = base / slugify(wk["week_title"])

        readings = [i for i in wk["items"] if i["type"] == "reading"]
        for it in (i for i in wk["items"] if i["type"] == "video"):
            make_nb(week_folder, it, readings)

    print("Done. Notebooks created under:", base)

if __name__ == "__main__":
    from pathlib import Path
    main()

# UT Austin NLP Notebooks

This repository scaffolds a reproducible set of Jupyter notebooks that accompany the **UT Austin CS388/AI388/DSC395T: Natural Language Processing (Online MS)** course. Each notebook links to the official lecture videos and readings and includes runnable demonstrations and exercises.

## Getting started

1. Create and activate a Python 3.11 environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Scrape the official materials index and generate notebooks:
   ```bash
   python scripts/scrape_ut_course.py --out course_index.json
   python scripts/build_notebooks.py --index course_index.json --out notebooks/
   ```
4. Launch JupyterLab to explore the notebooks:
   ```bash
   jupyter lab
   ```

Heavy demos and training loops are guarded by the `CI` environment variable so that continuous integration runs remain fast. Set `CI=true` to skip long-running cells when executing notebooks programmatically (e.g., via `nbclient`).

## Attribution

The notebook index, video links, and reading list are sourced from the official UT Austin course page: [Lecture Videos and Readings](https://www.cs.utexas.edu/~gdurrett/courses/online-course/materials.html). We only link to external materials and do not redistribute course-owned content.

## Repository layout

```
.
├── assignments/
├── data/
├── notebooks/
├── scripts/
├── requirements.txt
└── README.md
```

Assignments folders contain public-friendly equivalents of the original coursework. The `scripts/` directory provides utilities for scraping the materials page, generating notebooks, and collecting reusable demo snippets.

## Continuous integration

The GitHub Actions workflow executes a small smoke-test subset of notebooks with `CI=true` to ensure that templated cells run without modification. Use `pre-commit run -a` locally to apply the repository's formatting and linting rules before opening pull requests.


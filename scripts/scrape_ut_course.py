import json
import re
from pathlib import Path

import requests
from bs4 import BeautifulSoup

URL = "https://www.cs.utexas.edu/~gdurrett/courses/online-course/materials.html"


def clean(text: str) -> str:
    """Collapse whitespace and strip leading/trailing spaces."""
    return re.sub(r"\s+", " ", text).strip()


def main() -> None:
    html = requests.get(URL, timeout=30).text
    soup = BeautifulSoup(html, "html.parser")

    weeks = []
    current = None

    for el in soup.find_all(string=re.compile(r"^Week\s+\d+:", re.I)):
        if current:
            weeks.append(current)
        current = {"week_title": clean(el), "items": []}
        node = el.parent
        while node and not (
            node.name in ("h2", "h3")
            and node.get_text(strip=True).startswith("Week")
            and node.get_text(strip=True) != clean(el)
        ):
            for link in node.find_all("a", href=True):
                url = link["href"]
                item_type = "video" if ("youtu" in url) else "reading"
                current["items"].append(
                    {"type": item_type, "title": clean(link.get_text()), "url": url}
                )
            node = node.find_next_sibling()
    if current:
        weeks.append(current)

    Path("course_index.json").write_text(
        json.dumps({"source": URL, "weeks": weeks}, indent=2), encoding="utf-8"
    )
    print("Wrote course_index.json:", sum(len(w["items"]) for w in weeks), "items")


if __name__ == "__main__":
    main()

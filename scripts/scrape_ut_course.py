import json
import re
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup, Tag

URL = "https://www.cs.utexas.edu/~gdurrett/courses/online-course/materials.html"
WEEK_HEADER = re.compile(r"^Week\s+\d+:", re.I)


def clean(text: str) -> str:
    """Collapse whitespace and strip leading/trailing spaces."""
    return re.sub(r"\s+", " ", text).strip()


def scrape() -> dict:
    html = requests.get(URL, timeout=30).text
    soup = BeautifulSoup(html, "html.parser")

    # Locate the "Lecture Videos and Readings" section table.
    lectures_heading = None
    for tag in soup.find_all(["h2", "h3"]):
        if "Lecture Videos and Readings" in tag.get_text():
            lectures_heading = tag
            break
    if lectures_heading is None:
        raise RuntimeError("Couldn't locate the 'Lecture Videos and Readings' section")

    table = lectures_heading.find_next("table")
    if table is None:
        raise RuntimeError("Couldn't locate the course content table after the heading")

    weeks = []
    current = None

    for row in table.find_all("tr"):
        text = clean(row.get_text(" ", strip=True))
        if WEEK_HEADER.match(text):
            current = {"week_title": text, "items": []}
            weeks.append(current)
            continue

        if current is None:
            continue

        cells = row.find_all("td")
        if not cells:
            continue

        # First column contains videos/topics.
        for link in cells[0].find_all("a", href=True):
            title = clean(link.get_text())
            if not title:
                continue
            url = urljoin(URL, link["href"])
            item_type = "video" if "youtu" in url else "reading"
            current["items"].append({"type": item_type, "title": title, "url": url})

        # Second column (if present) contains readings.
        if len(cells) > 1:
            for link in cells[1].find_all("a", href=True):
                title = clean(link.get_text())
                if not title:
                    continue
                url = urljoin(URL, link["href"])
                current["items"].append({"type": "reading", "title": title, "url": url})

    assignments = []
    for tag in soup.find_all(["h2", "h3"]):
        if tag.get_text(strip=True).lower().startswith("assignments"):
            sibling = tag.next_sibling
            while sibling and not (isinstance(sibling, Tag) and sibling.name in ("h2", "h3")):
                if isinstance(sibling, Tag):
                    for link in sibling.find_all("a", href=True):
                        assignments.append(
                            {"title": clean(link.get_text()), "url": urljoin(URL, link["href"])}
                        )
                sibling = sibling.next_sibling
            break

    return {"source": URL, "weeks": weeks, "assignments": assignments}


def main() -> None:
    index = scrape()
    Path("course_index.json").write_text(
        json.dumps(index, indent=2), encoding="utf-8"
    )
    total_items = sum(len(week["items"]) for week in index["weeks"])
    print(
        f"Wrote course_index.json with {len(index['weeks'])} weeks, "
        f"{total_items} items, and {len(index.get('assignments', []))} assignments"
    )


if __name__ == "__main__":
    main()

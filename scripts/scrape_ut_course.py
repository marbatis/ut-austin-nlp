"""Scrape the UT Austin NLP online course materials page into JSON."""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from typing import List

import requests
from bs4 import BeautifulSoup

URL = "https://www.cs.utexas.edu/~gdurrett/courses/online-course/materials.html"


@dataclass
class Item:
    type: str
    title: str
    url: str
    notes: str | None = None


@dataclass
class Week:
    week_id: str
    week_title: str
    items: List[Item] = field(default_factory=list)


@dataclass
class CourseIndex:
    source: str
    weeks: List[Week]
    assignments: List[Item]


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _guess_type(url: str) -> str:
    if "youtu" in url:
        return "video"
    if any(ext in url for ext in (".pdf", "arxiv", "aclweb", "doi")):
        return "reading"
    return "link"


def fetch_html(timeout: int = 30) -> str:
    response = requests.get(URL, timeout=timeout)
    response.raise_for_status()
    return response.text


def parse_html(html: str) -> CourseIndex:
    soup = BeautifulSoup(html, "html.parser")

    weeks: List[Week] = []
    heading_regex = re.compile(r"^Week\s+\d+")
    for header in soup.find_all(["h2", "h3", "strong"], string=heading_regex):
        week_title = _clean(header.get_text())
        match = re.match(r"Week\s+(\d+)", week_title)
        week_id = match.group(1) if match else week_title

        items: List[Item] = []
        sibling = header.find_next_sibling()
        while sibling and not heading_regex.match(_clean(sibling.get_text())):
            links = sibling.find_all("a", href=True)
            for link in links:
                title = _clean(link.get_text()) or link.get("href", "").strip()
                url = link["href"].strip()
                if not url:
                    continue
                item_type = _guess_type(url)
                notes = None
                parent_text = _clean(sibling.get_text())
                if parent_text and parent_text != title:
                    notes = parent_text
                items.append(Item(type=item_type, title=title, url=url, notes=notes))
            sibling = sibling.find_next_sibling()
            if sibling and sibling.name == "hr":
                sibling = sibling.find_next_sibling()
        weeks.append(Week(week_id=week_id, week_title=week_title, items=items))

    assignments: List[Item] = []
    assignment_header = soup.find(string=re.compile(r"Assignments", re.I))
    if assignment_header:
        section = assignment_header.parent
        sibling = section.find_next_sibling()
        while sibling:
            links = sibling.find_all("a", href=True)
            if not links:
                sibling = sibling.find_next_sibling()
                continue
            for link in links:
                assignments.append(
                    Item(
                        type="assignment",
                        title=_clean(link.get_text()) or link["href"],
                        url=link["href"],
                    )
                )
            sibling = sibling.find_next_sibling()
            if sibling and sibling.name in {"h2", "h3"}:
                break

    return CourseIndex(source=URL, weeks=weeks, assignments=assignments)


def course_index_to_dict(index: CourseIndex) -> dict:
    return {
        "source": index.source,
        "weeks": [
            {
                "week_id": week.week_id,
                "week_title": week.week_title,
                "items": [
                    {
                        "type": item.type,
                        "title": item.title,
                        "url": item.url,
                        **({"notes": item.notes} if item.notes else {}),
                    }
                    for item in week.items
                ],
            }
            for week in index.weeks
        ],
        "assignments": [
            {"title": assignment.title, "url": assignment.url}
            for assignment in index.assignments
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="course_index.json", help="Output JSON path")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON")
    args = parser.parse_args()

    html = fetch_html()
    index = parse_html(html)
    data = course_index_to_dict(index)

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2 if args.pretty else None)
    print(f"Wrote {args.out} with {len(index.weeks)} weeks and {len(index.assignments)} assignments")


if __name__ == "__main__":
    main()

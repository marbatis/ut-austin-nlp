import json, re
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup, Tag

URL = "https://www.cs.utexas.edu/~gdurrett/courses/online-course/materials.html"

WEEK_HDR = re.compile(r"^Week\s+\d+:", re.I)

def clean(x: str) -> str:
    return re.sub(r"\s+", " ", x).strip()

def scrape() -> dict:
    html = requests.get(URL, timeout=30).text
    soup = BeautifulSoup(html, "html.parser")

    # Find the "Lecture Videos and Readings" section
    lec = None
    for h2 in soup.find_all(["h2", "h3"]):
        if "Lecture Videos and Readings" in h2.get_text():
            lec = h2
            break
    if lec is None:
        raise RuntimeError("Couldn't locate 'Lecture Videos and Readings' section")

    table = lec.find_next("table")
    if table is None:
        raise RuntimeError("Couldn't locate course content table after the heading")

    weeks = []
    cur = None
    for row in table.find_all("tr"):
        text = clean(row.get_text(" ", strip=True))
        if WEEK_HDR.match(text):
            cur = {"week_title": clean(text), "items": []}
            weeks.append(cur)
            continue

        if not cur:
            continue

        cells = row.find_all("td")
        if not cells:
            continue

        video_cell = cells[0]
        for a in video_cell.find_all("a", href=True):
            title = clean(a.get_text())
            if not title:
                continue
            url = urljoin(URL, a["href"])
            typ = "video" if ("youtu.be" in url or "youtube.com" in url) else "reading"
            cur["items"].append({"type": typ, "title": title, "url": url})

        if len(cells) > 1:
            reading_cell = cells[1]
            for a in reading_cell.find_all("a", href=True):
                title = clean(a.get_text())
                if not title:
                    continue
                url = urljoin(URL, a["href"])
                cur["items"].append({"type": "reading", "title": title, "url": url})

    # Assignments (top of page)
    assignments = []
    assign_hdr = None
    for h2 in soup.find_all(["h2", "h3"]):
        if h2.get_text(strip=True).lower().startswith("assignments"):
            assign_hdr = h2
            break
    if assign_hdr:
        sib = assign_hdr.next_sibling
        while sib and not (isinstance(sib, Tag) and sib.name in ("h2", "h3")):
            if isinstance(sib, Tag):
                for a in sib.find_all("a", href=True):
                    assignments.append({"title": clean(a.get_text()), "url": urljoin(URL, a["href"])})
            sib = sib.next_sibling

    return {"source": URL, "weeks": weeks, "assignments": assignments}

if __name__ == "__main__":
    out = Path("course_index.json")
    idx = scrape()
    out.write_text(json.dumps(idx, indent=2), encoding="utf-8")
    print(f"Wrote {out} with {len(idx['weeks'])} weeks and {len(idx.get('assignments', []))} assignments")

#!/usr/bin/env python3
"""Refresh READINGS.md from the course page using only Python's standard library."""

import argparse
from datetime import date
from html.parser import HTMLParser
from pathlib import Path
import re
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen


SOURCE_URL = "https://www.cs.columbia.edu/~blei/fogm/2026F/readings.html"
OUTPUT = Path(__file__).resolve().parents[1] / "READINGS.md"


def normalize(text):
    return " ".join(text.split())


class ReadingParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.lists = []
        self.topics = []
        self.topic = None
        self.reading = None
        self.paragraph = None
        self.paragraphs = []

    def handle_starttag(self, tag, attrs):
        if tag in {"ol", "ul"}:
            self.lists.append(tag)
        elif tag == "p":
            self.paragraph = []
        elif tag == "li" and self.lists == ["ol"]:
            self.topic = {"title": [], "readings": []}
            self.topics.append(self.topic)
        elif tag == "li" and self.lists == ["ol", "ul"]:
            self.reading = {"text": [], "links": []}
            self.topic["readings"].append(self.reading)
        elif tag == "a" and self.reading is not None:
            href = dict(attrs).get("href")
            if href:
                self.reading["links"].append(urljoin(SOURCE_URL, href))

    def handle_endtag(self, tag):
        if tag in {"ol", "ul"}:
            if not self.lists or self.lists.pop() != tag:
                raise ValueError("The course page's list structure has changed.")
        elif tag == "li" and self.lists == ["ol", "ul"]:
            self.reading = None
        elif tag == "li" and self.lists == ["ol"]:
            self.topic = None
        elif tag == "p" and self.paragraph is not None:
            self.paragraphs.append(normalize("".join(self.paragraph)))
            self.paragraph = None

    def handle_data(self, data):
        if self.paragraph is not None:
            self.paragraph.append(data)
        if self.reading is not None:
            self.reading["text"].append(data)
        elif self.topic is not None and self.lists == ["ol"]:
            self.topic["title"].append(data)


def render(html):
    page = ReadingParser()
    page.feed(html)
    page.close()
    if page.lists or not page.topics:
        raise ValueError("No complete reading list found; existing index was not changed.")

    main = next((p for p in page.paragraphs if "The main text is" in p), "")
    book = re.search(r'"([^"]+)" by (.+?)\.(?:\s+We\b|$)', main)
    if not book:
        raise ValueError("Main textbook details could not be read; review the source page.")

    total = sum(len(t["readings"]) for t in page.topics)
    linked = sum(bool(r["links"]) for t in page.topics for r in t["readings"])
    lines = [
        "# Course readings", "",
        f"Source: [Topics and readings — Fall 2026]({SOURCE_URL})  ",
        f"Last refreshed: {date.today().isoformat()}  ",
        f"{total} readings · {len(page.topics)} topics · {linked} linked entries · "
        f"{total - linked} entries without a supplied link", "",
        "The topics below follow the course page's order; their numbers are not a dated weekly schedule. "
        "Titles, author/year labels, and chapter details follow the course page. "
        "Where the page says “relevant chapters,” it does not specify chapter numbers.", "",
        "## Main textbook", "",
        f"**{book.group(1)}** — {book.group(2)}.", "",
    ]
    if "draft to enrolled" in main:
        lines += ["The course provides enrolled students with a draft; the reading page supplies no download link.", ""]

    lines += ["## Topics", ""]
    for number, topic in enumerate(page.topics, 1):
        title = normalize("".join(topic["title"]))
        if not title or not topic["readings"]:
            raise ValueError("An empty topic was found; review the course page before refreshing.")
        lines.append(f"{number}. [{title}](#topic-{number:02d})")

    for number, topic in enumerate(page.topics, 1):
        title = normalize("".join(topic["title"]))
        lines += ["", f'<a id="topic-{number:02d}"></a>', "", f"## {number}. {title}", ""]
        for reading in topic["readings"]:
            text = normalize("".join(reading["text"]))
            match = re.fullmatch(r'"(.+?)"\s*(.*)', text)
            if not match or len(reading["links"]) > 1:
                raise ValueError(f"Unexpected reading format: {text!r}")
            name, details = match.groups()
            name = name.replace("[", r"\[").replace("]", r"\]")
            if reading["links"]:
                url = reading["links"][0]
                if urlparse(url).scheme not in {"http", "https"}:
                    raise ValueError(f"Unexpected link scheme: {url!r}")
                label = f"[{name}](<{url}>)"
                suffix = ""
            else:
                label = f"**{name}**"
                suffix = " — No link supplied on the course page."
            lines.append(f"- {label} {details}{suffix}")

    lines += ["", "## Refresh this index", "",
        "From the repository root, run:", "", "```bash",
        "python3 scripts/update_readings.py", "```", "",
        "This file is generated from the course page. Keep your own notes and reading progress in "
        "separate files so a refresh preserves them. Refreshing retrieves the list; it does not "
        "download the readings or check whether each destination is accessible.", ""]
    return "\n".join(lines), total, len(page.topics)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--html", type=Path, help="Use a saved source page instead of fetching it.")
    args = parser.parse_args()
    if args.html:
        html = args.html.read_text(encoding="utf-8")
    else:
        request = Request(SOURCE_URL, headers={"User-Agent": "CourseReadingIndex/1.0"})
        with urlopen(request, timeout=30) as response:
            html = response.read().decode(response.headers.get_content_charset() or "utf-8")
    markdown, total, topics = render(html)
    OUTPUT.write_text(markdown, encoding="utf-8")
    print(f"Updated {OUTPUT.name}: {total} readings across {topics} topics.")


if __name__ == "__main__":
    main()

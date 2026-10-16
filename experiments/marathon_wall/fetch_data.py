"""Fetch the Boston Marathon split files the analysis runs on.

The primary files are a public third-party scrape of BAA results, pinned to a
source commit and checked against known SHA-256 hashes.  Optional 2015--2017
Kaggle files are loaded separately if the user supplies them.  This downloader
does not assume that a third-party scrape contains every official finisher.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Final, Mapping, Sequence

# llimllib/bostonmarathon: a scrape of baa.org.  2013 and 2014 are the only
# years in that repository carrying per-checkpoint splits.
_SOURCE_COMMIT: Final[str] = "f49b44950cdd13fdeaf1661ba921edbe3dfabff7"
_LLIMLLIB: Final[str] = (
    "https://raw.githubusercontent.com/llimllib/bostonmarathon/"
    + _SOURCE_COMMIT + "/results/{year}/results.csv"
)
_FETCHABLE: Final[Mapping[int, str]] = {year: _LLIMLLIB.format(year=year) for year in (2013, 2014)}
_SHA256: Final[Mapping[int, str]] = {
    2013: "28c2b1e580895cadab4f9b92893e9e4211b178ccbe374ae987581353c9963408",
    2014: "4ff8d8f25344f358abe70faba8e901aba703e964aff2da676b0ba0092f247437",
}

# Place these by hand from https://www.kaggle.com/datasets/rojour/boston-results
_MANUAL: Final[tuple[str, ...]] = (
    "marathon_results_2015.csv",
    "marathon_results_2016.csv",
    "marathon_results_2017.csv",
)


def _atomic_write(target: Path, content: bytes) -> None:
    """Replace a file only after its complete contents have been written."""
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=f".{target.name}.", delete=False) as handle:
            temporary = handle.name
            handle.write(content)
        os.replace(temporary, target)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)


def fetch(data_dir: Path, *, overwrite: bool = False, timeout: float = 180.0) -> list[Path]:
    """Download/verify the pinned files and write a reproducibility manifest."""
    data_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    manifest: list[dict[str, object]] = []
    for year, url in sorted(_FETCHABLE.items()):
        target = data_dir / f"results_{year}.csv"
        if target.exists() and not overwrite:
            content = target.read_bytes()
        else:
            print(f"[get ] {url}")
            with urllib.request.urlopen(url, timeout=timeout) as response:
                content = response.read()
        digest = hashlib.sha256(content).hexdigest()
        if digest != _SHA256[year]:
            raise OSError(
                f"{target}: SHA-256 mismatch; expected {_SHA256[year]}, got {digest}. "
                "Check the source or use --overwrite to replace an altered local copy."
            )
        if not target.exists() or overwrite:
            _atomic_write(target, content)
            written.append(target)
        print(f"[ok  ] {target} ({len(content) / 1e6:.1f} MB; SHA-256 verified)")
        manifest.append({
            "year": year, "filename": target.name, "source_url": url,
            "source_commit": _SOURCE_COMMIT, "sha256": digest,
            "bytes": len(content), "schema": "llimllib",
        })
    _atomic_write(data_dir / "source_manifest.json", (json.dumps({
        "source_repository": "https://github.com/llimllib/bostonmarathon",
        "collection_notes": "Third-party BAA scrape; finishers only; incomplete coverage; times rounded to 0.01 minutes.",
        "files": manifest,
    }, indent=2) + "\n").encode("utf-8"))
    return written


def report_status(data_dir: Path) -> None:
    """Say which of the optional 2015--2017 files are still missing."""
    missing = [name for name in _MANUAL if not (data_dir / name).exists()]
    if not missing:
        return
    print(
        "\nOptional: the 2015-2017 races add the 15K mat and a real thermal contrast\n"
        "(2015 cold and wet, 2017 hot).  This command does not fetch them; obtain\n"
        "them separately from\n"
        "    https://www.kaggle.com/datasets/rojour/boston-results\n"
        f"and drop these into {data_dir}/ -- the loader picks them up automatically:\n"
        + "".join(f"    {name}\n" for name in missing)
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("data/boston"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        fetch(args.data_dir, overwrite=args.overwrite)
    except OSError as error:
        print(f"download failed: {error}", file=sys.stderr)
        return 1
    report_status(args.data_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Fetch the Boston Marathon split files the analysis runs on.

The BAA does not publish a bulk download, so two public mirrors of its results
feed are used.  Only the 2013--2014 mirror is fetchable without credentials;
the 2015--2017 files live behind a Kaggle login and must be placed by hand,
which :func:`report_status` explains if they are absent.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path
from typing import Final, Mapping, Sequence

# llimllib/bostonmarathon: a scrape of baa.org.  2013 and 2014 are the only
# years in that repository carrying per-checkpoint splits.
_LLIMLLIB: Final[str] = "https://raw.githubusercontent.com/llimllib/bostonmarathon/master/results/{year}/results.csv"
_FETCHABLE: Final[Mapping[int, str]] = {year: _LLIMLLIB.format(year=year) for year in (2013, 2014)}

# Place these by hand from https://www.kaggle.com/datasets/rojour/boston-results
_MANUAL: Final[tuple[str, ...]] = (
    "marathon_results_2015.csv",
    "marathon_results_2016.csv",
    "marathon_results_2017.csv",
)


def fetch(data_dir: Path, *, overwrite: bool = False, timeout: float = 180.0) -> list[Path]:
    """Download the fetchable race files into ``data_dir``."""
    data_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for year, url in sorted(_FETCHABLE.items()):
        target = data_dir / f"results_{year}.csv"
        if target.exists() and not overwrite:
            print(f"[skip] {target} already present")
            continue
        print(f"[get ] {url}")
        with urllib.request.urlopen(url, timeout=timeout) as response:
            target.write_bytes(response.read())
        print(f"[ok  ] {target} ({target.stat().st_size / 1e6:.1f} MB)")
        written.append(target)
    return written


def report_status(data_dir: Path) -> None:
    """Say which of the optional 2015--2017 files are still missing."""
    missing = [name for name in _MANUAL if not (data_dir / name).exists()]
    if not missing:
        return
    print(
        "\nOptional: the 2015-2017 races add the 15K mat and a real thermal contrast\n"
        "(2015 cold and wet, 2017 hot).  They are not downloadable without a Kaggle\n"
        "login; download them from\n"
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

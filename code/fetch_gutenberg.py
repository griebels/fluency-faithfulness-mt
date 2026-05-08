#!/usr/bin/env python3
"""
Download and clean Project Gutenberg texts by ID.

Reads Gutenberg volume IDs from a metadata CSV, fetches each text, strips
Project Gutenberg headers/footers, and saves clean .txt files to an output folder.

Usage:
    python fetch_gutenberg.py --metadata metadata.csv --output_dir ./gutenberg_txt
"""

import argparse
import sys
import time
from pathlib import Path

import gutenbergpy.textget as tg
import pandas as pd
import requests



def fetch_pg_bytes(gid, timeout: int = 20):
    """
    Fetch the raw bytes of a Gutenberg text by ID.

    Tries the gutenbergpy library first (fast when it works), then falls back
    to a set of direct URL patterns that cover most titles.
    """
    # Try the library first
    try:
        raw = tg.get_text_by_id(gid)
        if raw and len(raw) > 200:
            return raw
    except TypeError:
        # gutenbergpy has a known bug where it raises TypeError ('raise None')
        # for some IDs — fall through to direct download
        pass
    except Exception:
        pass

    # Direct URL fallback — covers the most common Gutenberg file layouts
    candidates = [
        f"https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt",
        f"https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt.utf8",
        f"https://www.gutenberg.org/ebooks/{gid}.txt.utf-8",
        f"https://www.gutenberg.org/ebooks/{gid}.txt.utf8",
        f"https://www.gutenberg.org/cache/epub/{gid}/pg{gid}-0.txt",
        f"https://www.gutenberg.org/cache/epub/{gid}/pg{gid}-0.txt.utf8",
    ]

    last_err = None
    for url in candidates:
        try:
            r = requests.get(url, timeout=timeout)
            if r.status_code == 200 and r.content and len(r.content) > 200:
                return r.content
        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(f"Could not fetch PG text for id {gid}. Last error: {last_err}")


def fetch_and_save(gid, output_dir):
    """Fetch, strip headers using gutenbergpy script, and write a 
    single Gutenberg text to disk."""
    raw = fetch_pg_bytes(gid)
    clean = tg.strip_headers(raw).decode("utf-8", errors="replace").strip()
    (output_dir / f"{gid}.txt").write_text(clean, encoding="utf-8")


def load_ids_from_csv(csv_path, id_column):
    """Read Gutenberg IDs from a metadata CSV, dropping nulls and duplicates."""
    df = pd.read_csv(csv_path, encoding="utf-8")
    if id_column not in df.columns:
        print(f"ERROR: column '{id_column}' not found in {csv_path.name}.", file=sys.stderr)
        print(f"Available columns: {list(df.columns)}", file=sys.stderr)
        sys.exit(1)
    ids = df[id_column].dropna().unique().tolist()
    return [str(i).strip() for i in ids]



def download_all(ids, output_dir, delay: float = 0.5):
    """
    Download all IDs, skipping ones already saved, and report failures.
    A short delay between requests is included to be polite to Gutenberg's servers.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    already_done = {p.stem for p in output_dir.glob("*.txt")}
    pending = [gid for gid in ids if str(gid) not in already_done]

    print(f"Total IDs  : {len(ids)}")
    print(f"Already saved: {len(already_done)}")
    print(f"To fetch   : {len(pending)}\n")

    failures = []

    for n, gid in enumerate(pending, 1):
        print(f"[{n}/{len(pending)}] Fetching {gid}...", end=" ", flush=True)
        try:
            fetch_and_save(gid, output_dir)
            print("OK")
        except Exception as e:
            print(f"FAILED — {e}")
            failures.append((gid, str(e)))

        if n < len(pending):
            time.sleep(delay)

    print(f"\nDone. Saved {len(pending) - len(failures)} / {len(pending)} files.")
    if failures:
        print(f"\nFailed IDs ({len(failures)} total):")
        for gid, err in failures:
            print(f"  - {gid}: {err}")



def parse_args():
    parser = argparse.ArgumentParser(
        description="Download Project Gutenberg texts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--metadata", type=Path, help="Metadata CSV with Gutenberg IDs")
    source.add_argument("--ids", nargs="+", help="Gutenberg ID(s) to fetch")

    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--id_column", default="Volume ID in Gutenberg")
    parser.add_argument("--delay", type=float, default=0.5)

    return parser.parse_args()


def main():
    args = parse_args()

    if args.metadata:
        ids = load_ids_from_csv(args.metadata, args.id_column)
    else:
        ids = [str(i).strip() for i in args.ids]

    print(f"Loaded {len(ids)} Gutenberg ID(s).")
    download_all(ids, args.output_dir, delay=args.delay)


if __name__ == "__main__":
    main()

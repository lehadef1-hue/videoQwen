#!/usr/bin/env python3
"""Fetch all star names from postyl API and save to a text file."""
import requests
import sys

API_URL = "https://models.postyl.com/api/sync_models/get_stars"
PAGE_SIZE = 500
OUTPUT_FILE = "postyl_names.txt"


def fetch_all_names() -> list[str]:
    names = []
    page = 1
    while True:
        resp = requests.get(API_URL, params={
            "from_date": "2000-01-01 00:00:00",
            "size": PAGE_SIZE,
            "page": page,
        }, timeout=30)
        if not resp.ok:
            print(f"Page {page}: server returned {resp.status_code}, stopping.")
            break
        data = resp.json()

        rows = data.get("data", [])
        if not rows:
            break

        for item in rows:
            title = (item.get("title") or "").strip()
            if title:
                names.append(title)

        print(f"Page {page}: got {len(rows)} entries (total so far: {len(names)})")
        if len(rows) < PAGE_SIZE:
            break
        page += 1

    return names


if __name__ == "__main__":
    print("Fetching star names from postyl API...")
    names = fetch_all_names()
    print(f"\nTotal names: {len(names)}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for name in names:
            f.write(name + "\n")

    print(f"Saved to {OUTPUT_FILE}")
    print(f"\nNow run:")
    print(f"  python build_performer_db.py --from-file {OUTPUT_FILE}")

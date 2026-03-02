"""
Build a performer face-embedding database from ThePornDB API.

Setup
-----
1. Register for a free API token at  https://theporndb.net/register
2. Set the env var:   export TPDB_API_TOKEN=your_token_here
   (or put it in a .env file and use python-dotenv)
3. Install deps:      pip install insightface onnxruntime-gpu requests pillow opencv-python-headless

Usage examples
--------------
# Add specific performers by name
python build_performer_db.py --names "Mia Khalifa" "Lana Rhoades" "Riley Reid"

# Add from a text file (one name per line)
python build_performer_db.py --from-file names.txt

# Auto-populate with the N most popular performers from TPDB
python build_performer_db.py --auto --count 200

# Use a custom DB path
python build_performer_db.py --names "Aria Lee" --db /data/my_performers.pkl

# Show current DB contents
python build_performer_db.py --list

# Remove a performer
python build_performer_db.py --remove "Wrong Name"
"""

import os
import sys
import io
import pickle
import logging
import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional

import requests
import numpy as np
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("db_builder")

# ── Config ────────────────────────────────────────────────────────────────────
TPDB_BASE  = "https://api.theporndb.net"
TPDB_TOKEN = os.getenv("TPDB_API_TOKEN", "")
DB_PATH    = os.getenv("PERFORMER_DB_PATH", "/workspace/my_performers.pkl")

# ── InsightFace ───────────────────────────────────────────────────────────────
_face_app = None


def _get_face_app():
    global _face_app
    if _face_app is not None:
        return _face_app
    try:
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(
            name="buffalo_l",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        app.prepare(ctx_id=0, det_size=(640, 640))
        _face_app = app
        logger.info("InsightFace loaded")
    except Exception as e:
        logger.error(f"InsightFace not available: {e}")
        sys.exit(1)
    return _face_app


def _face_embedding(img: Image.Image) -> Optional[np.ndarray]:
    import cv2
    app = _get_face_app()
    rgb = np.array(img.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    faces = app.get(bgr)
    if not faces:
        return None
    # Pick the largest face (most likely the performer's headshot)
    best = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    if best.embedding is None:
        return None
    v = best.embedding
    return v / (np.linalg.norm(v) + 1e-8)


# ── TPDB helpers ──────────────────────────────────────────────────────────────

def _headers() -> Dict:
    if not TPDB_TOKEN:
        logger.error("TPDB_API_TOKEN is not set. Register at https://theporndb.net/register")
        sys.exit(1)
    return {"Authorization": f"Bearer {TPDB_TOKEN}", "Accept": "application/json"}


def tpdb_search(name: str) -> Optional[Dict]:
    """Search TPDB for a performer by name; return best match or None."""
    try:
        r = requests.get(
            f"{TPDB_BASE}/performers",
            params={"q": name},
            headers=_headers(),
            timeout=15,
        )
        r.raise_for_status()
        results = r.json().get("data", [])
        if not results:
            logger.warning(f"  ✗ Not found on TPDB: {name!r}")
            return None
        return results[0]
    except requests.RequestException as e:
        logger.error(f"  TPDB request failed for {name!r}: {e}")
        return None


def tpdb_top_performers(count: int) -> List[Dict]:
    """Fetch top N performers sorted by popularity."""
    performers = []
    page = 1
    per_page = min(count, 100)
    while len(performers) < count:
        try:
            r = requests.get(
                f"{TPDB_BASE}/performers",
                params={"sort": "views", "per_page": per_page, "page": page},
                headers=_headers(),
                timeout=15,
            )
            r.raise_for_status()
            data = r.json()
            batch = data.get("data", [])
            if not batch:
                break
            performers.extend(batch)
            if len(data.get("data", [])) < per_page:
                break  # last page
            page += 1
            time.sleep(0.3)  # be polite to the API
        except requests.RequestException as e:
            logger.error(f"TPDB pagination error: {e}")
            break
    return performers[:count]


def download_image(url: str) -> Optional[Image.Image]:
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert("RGB")
    except Exception as e:
        logger.debug(f"  Image download failed ({url}): {e}")
        return None


# ── Database I/O ──────────────────────────────────────────────────────────────

def load_db(path: str) -> Dict[str, List[np.ndarray]]:
    p = Path(path)
    if not p.exists():
        return {}
    with open(p, "rb") as f:
        return pickle.load(f)


def save_db(db: Dict[str, List[np.ndarray]], path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(db, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(f"Saved DB → {path}  ({len(db)} performers)")


# ── Core: add one performer ───────────────────────────────────────────────────

def add_performer(db: Dict, performer: Dict) -> bool:
    """
    Download images for a TPDB performer dict, extract embeddings, add to DB.
    Returns True if at least one embedding was added.
    """
    name = performer.get("name", "Unknown")
    image_urls: List[str] = []

    if performer.get("image"):
        image_urls.append(performer["image"])
    for extra in (performer.get("extra_images") or []):
        image_urls.append(extra)

    embeddings = []
    for url in image_urls[:6]:   # max 6 photos per performer
        img = download_image(url)
        if img is None:
            continue
        emb = _face_embedding(img)
        if emb is not None:
            embeddings.append(emb)
        time.sleep(0.1)

    if embeddings:
        db[name] = embeddings
        logger.info(f"  ✓ {name}  ({len(embeddings)} embedding(s))")
        return True
    else:
        logger.warning(f"  ✗ {name}: no face detected in any image")
        return False


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build performer face-embedding DB from ThePornDB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--names",     nargs="+", metavar="NAME", help="Performer name(s) to add")
    parser.add_argument("--from-file", metavar="FILE", help="Text file with one name per line")
    parser.add_argument("--auto",      action="store_true", help="Auto-populate from TPDB top performers")
    parser.add_argument("--count",     type=int, default=100, help="Number of performers for --auto (default: 100)")
    parser.add_argument("--remove",    nargs="+", metavar="NAME", help="Remove performer(s) from DB")
    parser.add_argument("--list",      action="store_true", help="List all performers in DB and exit")
    parser.add_argument("--db",        default=DB_PATH, help=f"DB file path (default: {DB_PATH})")
    args = parser.parse_args()

    db = load_db(args.db)
    logger.info(f"Existing DB: {len(db)} performer(s) in {args.db}")

    # ── --list ────────────────────────────────────────────────────────────────
    if args.list:
        if not db:
            print("Database is empty.")
        else:
            print(f"\n{'Performer':<40} {'Embeddings':>10}")
            print("-" * 52)
            for name, embs in sorted(db.items()):
                print(f"{name:<40} {len(embs):>10}")
            print(f"\nTotal: {len(db)} performer(s)")
        return

    # ── --remove ──────────────────────────────────────────────────────────────
    if args.remove:
        for name in args.remove:
            if name in db:
                del db[name]
                logger.info(f"Removed: {name}")
            else:
                logger.warning(f"Not in DB: {name}")
        save_db(db, args.db)
        return

    # ── Collect names ─────────────────────────────────────────────────────────
    names: List[str] = []
    if args.names:
        names.extend(args.names)
    if args.from_file:
        with open(args.from_file, encoding="utf-8") as f:
            names.extend(line.strip() for line in f if line.strip())

    if args.auto:
        logger.info(f"Fetching top {args.count} performers from TPDB…")
        top = tpdb_top_performers(args.count)
        for p in top:
            n = p.get("name", "")
            if n and n not in db:
                names.append(n)
        logger.info(f"Got {len(names)} new names from TPDB auto-populate")

    if not names:
        parser.error("Nothing to do. Use --names, --from-file, --auto, --list, or --remove.")

    # ── Process each name ─────────────────────────────────────────────────────
    added = 0
    for i, name in enumerate(names, 1):
        logger.info(f"[{i}/{len(names)}] {name}")
        if name in db:
            logger.info(f"  → already in DB, skipping")
            continue
        performer = tpdb_search(name)
        if performer:
            if add_performer(db, performer):
                added += 1
        # Save incrementally every 10 performers
        if i % 10 == 0:
            save_db(db, args.db)
        time.sleep(0.2)

    save_db(db, args.db)
    logger.info(f"Done. Added {added}/{len(names)} performer(s). DB now has {len(db)} entries.")


if __name__ == "__main__":
    main()

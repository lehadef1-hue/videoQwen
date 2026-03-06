"""
video_processor_v2.py — Advanced adult video analyzer with SEO output.
Runs on :8001 by default.
"""

import os, sys, cv2, base64, io, json, re, logging, shutil, subprocess, uuid, threading, queue
from dotenv import load_dotenv; load_dotenv()
from datetime import datetime
from pathlib import Path
from glob import glob
from typing import List, Dict, Optional, Tuple

import requests
from PIL import Image
from fastapi import FastAPI, Request, UploadFile, File, Form, Header, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ─── Performer recognition ────────────────────────────────────────────────────
try:
    from performer_finder import (
        detect_embeddings, cluster_embeddings, match_centroids,
        load_db as load_performer_db,
    )
    PERFORMER_RECOGNITION_AVAILABLE = True
except ImportError:
    PERFORMER_RECOGNITION_AVAILABLE = False

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(name)s | %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("analyzer_v2")

# ─── Config ───────────────────────────────────────────────────────────────────
MODEL_SERVER_URL  = os.getenv("MODEL_SERVER_URL",  "http://localhost:8080/generate")
UPLOAD_DIR        = os.getenv("UPLOAD_DIR",        "/workspace/video/videos")
OUTPUT_DIR        = os.getenv("OUTPUT_DIR",        "/workspace/video/result")
PERFORMER_DB_PATH = os.getenv("PERFORMER_DB_PATH", "/workspace/my_performers.pkl")
API_KEY           = os.getenv("API_KEY", "")          # empty = auth disabled

ALLOWED_CATEGORIES = [
    '3d Animation', 'Amateur', 'Anal', 'Anime', 'Arab', 'Asian', 'BBW', 'BDSM',
    'Bathrooms', 'Beach', 'Big Ass', 'Big Dick', 'Big Tits', 'Bisexual Male',
    'Blonde', 'Bondage', 'Boots', 'Brunette', 'Cartoon', 'Casting', 'Chubby',
    'Clamps', 'Cosplay', 'Creampie', 'Crossdresser', 'Cuckold', 'Cumshot',
    'Deep Throat', 'Device Bondage', 'Diaper', 'Double Penetration', 'Ebony',
    'Electricity Play', 'Facial', 'Fantasy Character', 'Femdom', 'Fetish',
    'Fisting', 'Foot Fetish', 'Furry', 'Futanari', 'Gags', 'Game Character',
    'Gameplay Video', 'Gangbang', 'Granny', 'Group', 'Gym', 'HD', 'Hairy',
    'Handcuffs', 'Handjob', 'Hanging Up', 'Hentai', 'Indian', 'Insect Fetish',
    'Interracial', 'Japanese Censored', 'Japanese Uncensored', 'Lactating',
    'Latex', 'Lesbian', 'Lingerie', 'MILF', 'Maid Uniform', 'Massage', 'Mature',
    'Midget', 'Nurse Uniform', 'Office', 'Oiled', 'Old/Young', 'OnlyFans',
    'Outdoor', 'POV', 'Pegging', 'Pissing', 'Pornstar', 'Pregnant', 'Public',
    'Pumping toy', 'Pussy Licking', 'Red Head', 'Rimjob', 'School Uniform',
    'Sensor Deprivation', 'Shibari Bondage', 'Shower', 'Slave Cage', 'Smoking',
    'Snuffing', 'Soap Play', 'Solo', 'Spanking', 'Sport', 'Squirt', 'Stockings',
    'Strap On', 'Tattoo', 'Teen', 'Tentacle', 'Thai', 'Threesome', 'Tickling',
    'Titty Fucking', 'Toilet', 'Toys', 'Transport Fetish', 'Vertical Video',
    'Vintage', 'Virtual Reality', 'Voyeur', 'Wax', 'Webcam', 'Whipping', 'Yoga',
]
ALLOWED_CATEGORIES_LOWER = {c.lower(): c for c in ALLOWED_CATEGORIES}
VALID_ORIENTATIONS = {"straight", "gay", "shemale"}

# ─── Content blocklist ────────────────────────────────────────────────────────
_CONTENT_BLOCKLIST = frozenset({
    "abused", "asphyxia", "behead", "bleed", "blood", "child", "children",
    "choke", "choking", "decapitation", "drugged", "forced", "hidden cam",
    "kid", "kill", "leaked", "loli", "l.o.l.i", "murder", "rape", "shota",
    "snuff", "strangle", "torture", "upskirt", "downblouse", "scat", "cp",
    "gore", "15yo", "16yo", "17yo",
})
_BLOCKED_RE = re.compile(
    "|".join(r"\b" + re.escape(w) + r"\b" for w in sorted(_CONTENT_BLOCKLIST, key=len, reverse=True)),
    re.IGNORECASE,
)

def _contains_blocked(text: str) -> bool:
    return bool(_BLOCKED_RE.search(text))

def _redact_blocked(text: str) -> str:
    return _BLOCKED_RE.sub("***", text)

def _filter_blocked_list(items: List[str]) -> List[str]:
    filtered = [item for item in items if not _contains_blocked(item)]
    if len(filtered) < len(items):
        logger.warning(f"Blocked {len(items) - len(filtered)} item(s) from list")
    return filtered


DESCRIPTION_STYLES = {
    "standard": (
        "Write a vivid, explicit, dirty description (3–5 sentences). "
        "Use raw, vulgar slang and dirty talk style. Describe performers, bodies, positions, actions in graphic detail. "
        "Write like a horny human — be crude, playful, nasty. Do NOT be clinical or polite. "
        "NEVER start with 'This video', 'In this video', or similar — jump straight into describing the action."
    ),
    "clean": (
        "Write a tasteful description (3–5 sentences) suitable for mainstream platforms. "
        "Focus on mood, setting, performers' appearance, and the nature of the encounter. "
        "Avoid explicit sexual terms. NEVER start with 'This video', 'In this video', or similar."
    ),
    "cinematic": (
        "Write a cinematic description (3–5 sentences) in the style of a film critic. "
        "Focus on visual composition, lighting quality, performers' chemistry, camera angles. "
        "Treat it as a review of visual and performative qualities. NEVER start with 'This video' or similar."
    ),
}

# ─── Prompts ──────────────────────────────────────────────────────────────────

def build_analysis_prompt(frame_count: int, ts_map: str, desc_style: str, language: str) -> str:
    return f"""You receive {frame_count} key frames sampled evenly across the full video.

FRAME TIMESTAMPS (seconds from video start):
{ts_map}

TASK: Analyze the video content and return structured metadata.

--- ORIENTATION ---
Choose EXACTLY ONE value: straight | gay | shemale
- straight  = male+female sex, OR all-female (lesbian = category tag, NOT an orientation)
- gay       = ONLY male performers having sex with each other — no women present at all
- shemale   = trans woman (MTF): visibly female body (breasts, feminine shape) WITH a penis visible

CRITICAL: "lesbian" is NOT a valid orientation — use "straight" instead.
CRITICAL: if you see a penis on a femininely-built performer → shemale, not straight.

--- DESCRIPTION ---
{desc_style}

--- CATEGORIES ---
RULES — apply ALL strictly. Tag ONLY what is CLEARLY and VISUALLY CONFIRMED:
Write 5–10 tags. HARD MAXIMUM: 10. Use standard adult site category names.

ACTS — only if clearly visible in multiple frames (not implied, not described):
  "Anal" = anal penetration by penis or toy, clearly in-frame
  "Double Penetration" = TWO simultaneous penetrations visible at the SAME TIME — NOT about performer count, purely about 2 orifices penetrated simultaneously
  "Fisting" = entire fist inside body, clearly visible — very rare, be strict
  "Squirt" = fluid ejaculation from vagina clearly shown
  "Creampie" = cum visibly dripping from vagina/anus after internal finish
  "Facial" = cum on face, clearly shown
  "Cumshot" = male ejaculation visible
  "Pussy Licking" = face-to-vulva oral contact clearly visible
  "Rimjob" = face-to-anus oral contact clearly visible
  "Deep Throat" = full deep-throat gag visible (not just blowjob)
  "Titty Fucking" = penis between breasts — REQUIRES a penis
  "Handjob" = hand on penis, clearly visible
  "Pegging" = woman penetrates man with strap-on

GROUPS — count only performers ACTIVELY PARTICIPATING in sex acts, NOT bystanders or people merely visible in frame:
  "Solo" = one performer actively performing alone throughout — no other participant at all
  "Threesome" = exactly 3 performers all actively participating in the same scene
  "Group" = 4+ performers all actively participating together
  "Gangbang" = 3+ performers actively on 1 — ALSO add "Group"
  "Lesbian" = female-female ONLY, all actively having sex — zero males participating
  "Bisexual Male" = one male ACTIVELY has sex with both a male AND a female in the same scene
  "Femdom" = female actively dominating, male visibly submissive — both must be active participants

APPEARANCE — only if unmistakably, obviously present:
  "Big Tits" / "Big Ass" / "Big Dick" / "BBW" = clearly larger than average
  "Tattoo" / "Hairy" / "Interracial" / "Teen" / hair color tags

FORBIDDEN COMBINATIONS (auto-removed in post-processing anyway, but still):
  NO "Titty Fucking" without a penis
  NO "Double Penetration" without two simultaneous penetrations visible by a penis
  NO "Gangbang" without "Group"
  NO "Solo" combined with group tags
WHEN IN DOUBT → OMIT.

--- WATERMARKS / ON-SCREEN TEXT ---
List ALL visible static text overlays, watermarks, logos across frames.
Return exact text as array (e.g. ["brazzers.com", "pornhub.com"]). Return [] if none.

--- KEY SCENES ---
Identify 5–8 key moments or scene changes using the frame timestamps above.
Reference frames by their 0-based index. Write a NEUTRAL 1-sentence description (no explicit terms).

--- OUTPUT ---
Return ONLY valid JSON, no markdown. All text fields in {language}.
{{"orientation":"straight","description":"...","watermarks":["site.com"],"categories":[...],"key_scenes":[{{"frame":2,"desc":"..."}}]}}"""


def build_seo_prompt(description: str, categories: List[str], orientation: str, language: str,
                     tag_count: int = 5, secondary_tag_count: int = 7) -> str:
    cats_str = ", ".join(categories)
    return f"""You are a professional SEO specialist for an adult content website. Your task is to generate fully optimized SEO metadata based on the provided video information.

VIDEO DESCRIPTION:
{description}

CATEGORIES: {cats_str}
ORIENTATION: {orientation}

GENERATE THE FOLLOWING:

1. META TITLE — 50–60 characters total.
   - Include the most important keyword phrase naturally.
   - Make it compelling and click-worthy. No ALL CAPS, no excessive punctuation.

2. META DESCRIPTION — 140–160 characters total.
   - Natural flowing sentence, not a keyword dump.
   - Include 2–3 relevant keyword phrases. Should entice users to click.

3. PRIMARY TAGS — exactly {tag_count} long-tail keyword phrases (3–6 words each).
   - Exact search queries real users type. Based on acts, appearance, setting.
   - Format: lowercase phrases, e.g. "amateur outdoor sex video"

4. SECONDARY TAGS — exactly {secondary_tag_count} shorter keyword phrases (2–4 words).
   - Broader supporting search terms. Mix acts, appearance, category keywords.
   - Format: lowercase phrases

5. SEO DESCRIPTION — 2–3 short paragraphs (80–120 words total).
   - Natural readable prose, not keyword stuffing. Start with main action and performers.
   - Mention key visual details, setting, mood. Tasteful language — no explicit profanity.
   - Keyword-rich but reads naturally.

--- OUTPUT ---
Return ONLY valid JSON, no markdown. All text in {language}.
{{"meta_title":"...","meta_description":"...","primary_tags":[...],"secondary_tags":[...],"seo_description":"..."}}"""


def build_seo_translate_prompt(meta_title: str, meta_desc: str, seo_description: str, language: str) -> str:
    return f"""Translate the following adult video SEO texts into {language}.
Rules:
- Keep meaning, tone, and keywords accurate.
- seo_description must stay under 150 words. Do NOT add new sentences. Do NOT repeat any phrase.
- Return ONLY valid JSON, no markdown, no extra text.

SOURCE TEXTS:
meta_title: {meta_title}
meta_description: {meta_desc}
seo_description: {seo_description}

{{"meta_title":"...","meta_description":"...","seo_description":"..."}}"""



# ─── Helpers ──────────────────────────────────────────────────────────────────

def _fmt_ts(seconds: float) -> str:
    s = int(seconds)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def pil_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=82, optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def call_vision_model(
    prompt: str,
    images: List[Image.Image],
    sampling: Optional[Dict] = None,
) -> str:
    payload = {
        "prompt": prompt,
        "base64_images": [pil_to_base64(img) for img in images],
        "sampling_params": sampling or {"temperature": 0.65, "top_p": 0.90, "max_tokens": 1200},
    }
    try:
        r = requests.post(MODEL_SERVER_URL, json=payload, timeout=420)
        r.raise_for_status()
        out = r.json().get("output", "").strip()
        logger.debug(f"Model response: {len(out)} chars")
        return out
    except Exception:
        logger.exception("Model call failed")
        return ""


def extract_json_from_response(text: str) -> Optional[Dict]:
    if not text:
        return None

    def _try_parse(s: str) -> Optional[Dict]:
        s = re.sub(r',\s*(?=[}\]])', '', s).strip()
        s = re.sub(r'(?<!\]),\s*"thumbnailIndex":\s*(\d+)\s*\}\s*$', r'], "thumbnailIndex": \1}', s)
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            return None

    for pattern in (r'```json\s*([\s\S]*?)```', r'```\s*([\s\S]*?)```'):
        m = re.search(pattern, text)
        if m:
            r = _try_parse(m.group(1))
            if r is not None:
                return r

    candidates = []
    depth, start = 0, None
    for i, ch in enumerate(text):
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and start is not None:
                candidates.append(text[start:i + 1])
                start = None

    for cand in sorted(candidates, key=len, reverse=True):
        r = _try_parse(cand)
        if r is not None:
            return r

    # Truncated JSON — append missing closing braces
    if depth > 0 and start is not None:
        truncated = text[start:]
        for suffix in ('}' * depth, '}' * depth + '}'):
            r = _try_parse(truncated + suffix)
            if r is not None:
                return r

    # Regex fallback for Pass 1 fields
    result: Dict = {}
    for key in ("orientation", "description"):
        m = re.search(rf'"{key}"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
        if m:
            result[key] = m.group(1)
    cats_m = re.search(r'"categories"\s*:\s*\[([^\]]*)', text)
    if cats_m:
        result["categories"] = re.findall(r'"((?:[^"\\]|\\.)*)"', cats_m.group(1))
    # Extract key_scenes objects individually
    scenes = []
    for scene_m in re.finditer(r'\{[^{}]*"frame"\s*:\s*(\d+)[^{}]*"desc"\s*:\s*"((?:[^"\\]|\\.)*)"[^{}]*\}', text):
        scenes.append({"frame": int(scene_m.group(1)), "desc": scene_m.group(2)})
    if not scenes:
        # also try desc before frame order
        for scene_m in re.finditer(r'\{[^{}]*"desc"\s*:\s*"((?:[^"\\]|\\.)*)"[^{}]*"frame"\s*:\s*(\d+)[^{}]*\}', text):
            scenes.append({"frame": int(scene_m.group(2)), "desc": scene_m.group(1)})
    if scenes:
        result["key_scenes"] = scenes
    if result:
        logger.warning("Used regex fallback for JSON")
        return result

    logger.error(f"JSON parse failed. Preview: {text[:400]}")
    return None


def extract_key_frames_ts(
    video_path: str,
    target_count: int = 25,
    start_at: Optional[int] = None,
    end_at: Optional[int] = None,
) -> Tuple[List[Image.Image], List[float]]:
    """Extract frames and their timestamps (seconds)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    if total < 8:
        cap.release()
        raise ValueError("Video too short")
    if start_at is None:
        start_at = max(1, int(total * 0.04))
    if end_at is None:
        end_at = total - max(1, int(total * 0.04))
    usable = max(1, end_at - start_at)
    step = max(1, usable // target_count)
    frames, timestamps = [], []
    for i in range(start_at, end_at, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        timestamps.append(float(i / fps))
        if len(frames) >= target_count:
            break
    cap.release()
    logger.info(f"Extracted {len(frames)} frames ({start_at}–{end_at}/{total})")
    return frames, timestamps


def extract_video_meta(video_path: str) -> Dict:
    """Extract container metadata + embedded cover art via ffprobe/ffmpeg."""
    meta: Dict = {}

    # ── Technical info & tags ───────────────────────────────────────────────
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_format", "-show_streams", video_path],
            capture_output=True, text=True, timeout=15,
        )
        if r.returncode == 0:
            data = json.loads(r.stdout)
            fmt  = data.get("format", {})
            tags = {k.lower(): v for k, v in (fmt.get("tags") or {}).items()}

            meta["duration"] = float(fmt.get("duration") or 0)
            meta["title"]    = tags.get("title", "").strip()
            meta["comment"]  = tags.get("comment", tags.get("description", "")).strip()
            meta["artist"]   = tags.get("artist", tags.get("author", "")).strip()

            # Resolution from first video stream
            for stream in data.get("streams", []):
                if stream.get("codec_type") == "video" and stream.get("codec_name") != "mjpeg":
                    w = int(stream.get("width") or 0)
                    h = int(stream.get("height") or 0)
                    if w and h:
                        meta["width"]    = w
                        meta["height"]   = h
                        meta["vertical"] = h > w
                        meta["hd"]       = (w >= 1280 or h >= 720)
                    break
    except Exception as e:
        logger.warning(f"ffprobe failed: {e}")

    # ── Embedded cover art ──────────────────────────────────────────────────
    meta["cover"] = None
    try:
        r2 = subprocess.run(
            ["ffmpeg", "-y", "-i", video_path,
             "-map", "0:v:1", "-vframes", "1", "-f", "image2pipe", "-vcodec", "mjpeg", "-"],
            capture_output=True, timeout=10,
        )
        if r2.returncode == 0 and r2.stdout:
            img = Image.open(io.BytesIO(r2.stdout))
            img.verify()
            img = Image.open(io.BytesIO(r2.stdout))
            meta["cover"] = img.convert("RGB")
            logger.info("Extracted embedded cover art")
    except Exception:
        pass  # No cover — normal

    return meta


def _normalize_cats(raw: list) -> List[str]:
    result, seen = [], set()
    for cat in raw:
        key = cat.lower().strip()
        if key in ALLOWED_CATEGORIES_LOWER and key not in seen:
            result.append(ALLOWED_CATEGORIES_LOWER[key])
            seen.add(key)
    return result


def validate_categories(categories: List[str], orientation: Optional[str]) -> List[str]:
    cats = list(categories)
    cats_lower = {c.lower() for c in cats}

    requires_penis = {"titty fucking", "handjob", "cumshot", "facial", "creampie", "big dick"}
    requires_vagina = {"pussy licking", "squirt", "creampie"}

    if orientation == "gay":
        forbidden = requires_vagina | {
            "lesbian", "milf", "squirt", "pregnant", "lactating",
            "pussy licking", "pegging", "strap on",
        }
        cats = [c for c in cats if c.lower() not in forbidden]

    elif orientation == "shemale":
        forbidden = {"lesbian", "gay", "pegging", "strap on", "pregnant", "lactating", "pussy licking", "squirt"}
        cats = [c for c in cats if c.lower() not in forbidden]

    elif orientation == "straight":
        cats = [c for c in cats if c.lower() != "gay"]
        if "lesbian" in cats_lower:
            cats = [c for c in cats if c.lower() not in requires_penis]

    cats_lower = {c.lower() for c in cats}

    if not any(x in cats_lower for x in ["group", "threesome", "gangbang"]):
        cats = [c for c in cats if c.lower() != "bisexual male"]

    if "gangbang" in cats_lower and "group" not in cats_lower:
        cats = [c for c in cats if c.lower() != "gangbang"]

    if "solo" in cats_lower:
        cats = [c for c in cats if c.lower() not in {
            "group", "threesome", "gangbang", "double penetration", "bisexual male", "lesbian"
        }]

    if orientation == "gay" or "lesbian" in {c.lower() for c in cats}:
        cats = [c for c in cats if c.lower() != "femdom"]

    return cats



def _seo_fallback(text: str) -> Dict:
    """Extract SEO fields from truncated/malformed JSON via regex."""
    result: Dict = {}
    for key in ("meta_title", "meta_description", "seo_description"):
        m = re.search(rf'"{key}"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
        if m:
            result[key] = m.group(1)
    for key in ("primary_tags", "secondary_tags"):
        m = re.search(rf'"{key}"\s*:\s*\[([^\]]*)', text)
        if m:
            items = re.findall(r'"((?:[^"\\]|\\.)*)"', m.group(1))
            if items:
                result[key] = items
    if result:
        logger.warning(f"SEO regex fallback used — recovered: {list(result.keys())}")
    return result


# ─── Core processor ───────────────────────────────────────────────────────────

def process_video_v2(
    video_path: str,
    output_dir: str,
    base_name: str,
    language: str = "English",
    style: str = "standard",
    extra_languages: Optional[List[str]] = None,   # additional lang codes ["de","es",...]
    tag_count: int = 5,
    secondary_tag_count: int = 7,
    category_count: int = 10,
) -> Dict:
    logger.info(f"Processing v2: {base_name} | lang={language} style={style}")
    try:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        skip4 = max(1, int(total_frames * 0.04))
        skip8 = max(1, int(total_frames * 0.08))
        end92 = total_frames - skip8

        # ── Pass 1: Analysis + key scenes ──────────────────────────────────────
        frames_1a, ts_1a = extract_key_frames_ts(video_path, 25, start_at=skip4, end_at=end92)
        if len(frames_1a) < 4:
            return {"status": "skipped", "reason": "too few frames"}

        ts_map = "  ".join(f"F{i}={_fmt_ts(t)}" for i, t in enumerate(ts_1a))
        desc_style = DESCRIPTION_STYLES.get(style, DESCRIPTION_STYLES["standard"])

        raw1 = call_vision_model(
            build_analysis_prompt(len(frames_1a), ts_map, desc_style, language),
            frames_1a,
            {"temperature": 0.45, "top_p": 0.85, "max_tokens": 2048},
        )
        p1 = extract_json_from_response(raw1)
        if not p1:
            return {"status": "error", "reason": "pass1 invalid response"}

        description = _redact_blocked(p1.get("description", "").strip())
        orientation = p1.get("orientation", "straight")
        if orientation not in VALID_ORIENTATIONS:
            orientation = "straight"
        cats_raw   = _filter_blocked_list(_normalize_cats(p1.get("categories", []))[:15])
        watermarks = [str(w).strip() for w in (p1.get("watermarks") or []) if str(w).strip()]

        # Convert key scene frame indices → timestamps
        key_scenes = []
        _seen_scene_descs: set = set()
        for ks in (p1.get("key_scenes") or [])[:8]:
            fi = ks.get("frame")
            if isinstance(fi, int) and 0 <= fi < len(ts_1a):
                desc = str(ks.get("desc", "")).strip()
                desc_key = desc.lower()[:80]
                if desc_key in _seen_scene_descs:
                    continue
                _seen_scene_descs.add(desc_key)
                key_scenes.append({
                    "ts": float(round(ts_1a[fi], 1)),
                    "formatted": _fmt_ts(ts_1a[fi]),
                    "desc": desc,
                })

        final_categories = validate_categories(cats_raw, orientation)
        logger.info(f"Pass 1: orient={orientation} cats={len(final_categories)} watermarks={watermarks} scenes={len(key_scenes)}")

        # Screenshot selection disabled
        top5 = []
        thumb_frame = None

        # ── Pass SEO (multi-language) ────────────────────────────────────────
        final_categories = final_categories[:category_count]
        seo_ref = [thumb_frame] if thumb_frame else frames_1a[:1]

        # Build list of (lang_code, lang_name) — first entry is primary language
        all_langs: List[Tuple[str, str]] = []
        # primary
        primary_code = next(
            (k for k, v in LANG_MAP.items() if v.lower() == language.lower()), "en"
        )
        all_langs.append((primary_code, language))
        # extras
        for code in (extra_languages or []):
            lc = code.lower()
            if lc in LANG_MAP and lc != primary_code:
                all_langs.append((lc, LANG_MAP[lc]))

        seo_by_lang: Dict[str, Dict] = {}

        # Primary language — full SEO (title + desc + tags)
        (p_lang_code, p_lang_name) = all_langs[0]
        raw_seo = call_vision_model(
            build_seo_prompt(description, final_categories, orientation, p_lang_name,
                             tag_count, secondary_tag_count),
            seo_ref,
            {"temperature": 0.3, "top_p": 0.85, "max_tokens": 4096},
        )
        p_seo = extract_json_from_response(raw_seo) or _seo_fallback(raw_seo or "")
        primary_tags   = _filter_blocked_list([t.strip() for t in p_seo.get("primary_tags", []) if t.strip()][:tag_count])
        secondary_tags = _filter_blocked_list([t.strip() for t in p_seo.get("secondary_tags", []) if t.strip()][:secondary_tag_count])
        seo_by_lang[p_lang_code] = {
            "meta_title":       _redact_blocked(p_seo.get("meta_title", "").strip()),
            "meta_description": _redact_blocked(p_seo.get("meta_description", "").strip()),
            "seo_description":  _redact_blocked(p_seo.get("seo_description", "").strip()),
        }
        logger.info(f"Pass SEO [{p_lang_code}]: title={len(seo_by_lang[p_lang_code]['meta_title'])}")

        # Extra languages — translate only meta_title, meta_description, seo_description
        base_title    = seo_by_lang[p_lang_code]["meta_title"]
        base_meta     = seo_by_lang[p_lang_code]["meta_description"]
        base_seo_desc = seo_by_lang[p_lang_code]["seo_description"]
        for lang_code, lang_name in all_langs[1:]:
            raw_tr = call_vision_model(
                build_seo_translate_prompt(base_title, base_meta, base_seo_desc, lang_name),
                seo_ref,
                {"temperature": 0.2, "top_p": 0.80, "max_tokens": 1024},
            )
            p_tr = extract_json_from_response(raw_tr) or {}
            seo_by_lang[lang_code] = {
                "meta_title":       _redact_blocked(p_tr.get("meta_title", "").strip()),
                "meta_description": _redact_blocked(p_tr.get("meta_description", "").strip()),
                "seo_description":  _redact_blocked(p_tr.get("seo_description", "").strip()),
            }
            logger.info(f"Pass SEO translate [{lang_code}]: title={len(seo_by_lang[lang_code]['meta_title'])}")

        # Flat fields from primary language for backward compat
        primary_seo     = seo_by_lang[p_lang_code]
        meta_title      = primary_seo["meta_title"]
        meta_desc       = primary_seo["meta_description"]
        seo_description = primary_seo["seo_description"]

        # ── Performer recognition ────────────────────────────────────────────────
        performers: List[Dict] = []
        if PERFORMER_RECOGNITION_AVAILABLE:
            try:
                frames_face, _ = extract_key_frames_ts(video_path, 100, start_at=skip4, end_at=end92)
                db = load_performer_db(PERFORMER_DB_PATH)
                if db:
                    centroids = cluster_embeddings(detect_embeddings(frames_face))
                    matches   = match_centroids(centroids, db)
                    performers = [{"name": m["name"], "score": round(m["score"] * 100)} for m in matches]
                    if performers:
                        logger.info(f"Performers: {performers}")
            except Exception as e:
                logger.warning(f"Performer detection failed: {e}")

        # ── Save results ────────────────────────────────────────────────────────
        os.makedirs(output_dir, exist_ok=True)
        saved_frames = []
        for i, cand in enumerate(top5):
            path = os.path.join(output_dir, f"{base_name}_frame_{i:03d}.jpg")
            cand["frame"].save(path, quality=85, optimize=True)
            saved_frames.append({
                "path": path,
                "score": cand["score"],
                "reason": cand.get("reason", ""),
                "ts": cand.get("ts"),
                "ts_fmt": cand.get("ts_fmt", ""),
            })

        thumb_path = None
        thumb_b64 = ""
        if thumb_frame is not None:
            thumb_path = os.path.join(output_dir, f"{base_name}_thumb.jpg")
            thumb_frame.save(thumb_path, quality=88, optimize=True)
            _buf = io.BytesIO()
            thumb_frame.save(_buf, format="JPEG", quality=85)
            thumb_b64 = base64.b64encode(_buf.getvalue()).decode()

        meta = {
            "description": description,
            "categories": final_categories,
            "orientation": orientation,
            "watermarks": watermarks,
            "performers": performers,
            "key_scenes": key_scenes,
            "seo": seo_by_lang,
            "meta_title": meta_title,
            "meta_description": meta_desc,
            "primary_tags": primary_tags,
            "secondary_tags": secondary_tags,
            "seo_description": seo_description,
            "language": language,
            "style": style,
        }
        with open(os.path.join(output_dir, f"{base_name}_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        def _to_url(p):
            if not p:
                return None
            try:
                rel = Path(p).relative_to(OUTPUT_DIR)
                return f"/v2/results/{rel.as_posix()}"
            except ValueError:
                return None

        logger.info(f"Done: {base_name}")
        return {
            "status": "ok",
            "base_name": base_name,
            "orientation": orientation,
            "description": description,
            "categories": final_categories,
            "watermarks": watermarks,
            "performers": performers,
            "key_scenes": key_scenes,
            "seo": seo_by_lang,
            "meta_title": meta_title,
            "meta_description": meta_desc,
            "primary_tags": primary_tags,
            "secondary_tags": secondary_tags,
            "seo_description": seo_description,
            "thumbnail": _to_url(thumb_path),
            "thumbnail_base64": thumb_b64,
            "frames": [
                {
                    "url": _to_url(f["path"]),
                    "score": f["score"],
                    "reason": f["reason"],
                    "ts": f["ts"],
                    "ts_fmt": f["ts_fmt"],
                }
                for f in saved_frames
            ],
        }

    except Exception as e:
        logger.exception(f"Critical error processing {video_path}")
        return {"status": "error", "reason": str(e)}


def _build_webhook_payload(task_id: str, result: Dict) -> Dict:
    """Convert internal result to the external webhook format."""
    if result.get("status") != "ok":
        return {"success": False, "task_id": task_id, "error": result.get("reason", "unknown")}

    seo = result.get("seo", {})
    primary_code = next(iter(seo), "en")

    performers_out = [
        {"name": p["name"], "confidence": p.get("score", 0)}
        for p in result.get("performers", [])
    ]

    r: Dict = {
        "primary_tags":     result.get("primary_tags", []),
        "secondary_tags":   result.get("secondary_tags", []),
        "categories":       result.get("categories", []),
        "orientation":      result.get("orientation", ""),
        "description":      result.get("description", ""),
        "watermarks":       result.get("watermarks", []),
        "performers":       performers_out,
        "meta_title":       result.get("meta_title", ""),
        "meta_description": result.get("meta_description", ""),
        "seo_description":  result.get("seo_description", ""),
        "preview_thumbnail": ("data:image/jpeg;base64," + result["thumbnail_base64"]) if result.get("thumbnail_base64") else "",
    }

    # Flatten extra languages: meta_title_de, meta_description_de, seo_description_de ...
    for lang_code, lang_data in seo.items():
        if lang_code == primary_code:
            continue
        r[f"meta_title_{lang_code}"]       = lang_data.get("meta_title", "")
        r[f"meta_description_{lang_code}"] = lang_data.get("meta_description", "")
        r[f"seo_description_{lang_code}"]  = lang_data.get("seo_description", "")

    return {"success": True, "task_id": task_id, "result": r}


# ─── Task store ───────────────────────────────────────────────────────────────
# {task_id: {"status": "processing"|"done"|"error", "stage": str, "result": dict}}
_tasks: Dict[str, Dict] = {}
_tasks_lock = threading.Lock()

# ─── Serial task queue ────────────────────────────────────────────────────────
_task_queue: queue.Queue = queue.Queue()

def _queue_worker():
    """Single worker — processes one video at a time, queues the rest."""
    while True:
        task_id, fn, args, kwargs, webhook_url = _task_queue.get()
        try:
            _run_task(task_id, fn, *args, webhook_url=webhook_url, **kwargs)
        except Exception as e:
            logger.error(f"Worker error {task_id}: {e}")
        finally:
            _task_queue.task_done()

threading.Thread(target=_queue_worker, daemon=True, name="task-worker").start()


def _run_task(task_id: str, fn, *args, webhook_url: str = "", **kwargs):
    """Run fn(*args, **kwargs) in a thread; store result in _tasks; fire webhook if set."""
    try:
        result = fn(*args, **kwargs)
    except Exception as e:
        result = {"status": "error", "reason": str(e)}
    with _tasks_lock:
        _tasks[task_id]["status"] = result.get("status", "error")
        _tasks[task_id]["result"] = result
    if webhook_url:
        try:
            payload = _build_webhook_payload(task_id, result)
            log_payload = {**payload, "result": {**payload.get("result", {}), "preview_thumbnail": f"<base64 {len(payload.get('result', {}).get('preview_thumbnail', ''))} chars>"}}
            logger.info(f"Webhook payload → {json.dumps(log_payload, ensure_ascii=False)}")
            wh = requests.post(webhook_url, json=payload, timeout=15,
                               headers={"User-Agent": _CHROME_UA, "Content-Type": "application/json"})
            logger.info(f"Webhook fired → {webhook_url} | status={wh.status_code} | response={wh.text[:5000]}")
        except Exception as e:
            logger.warning(f"Webhook failed: {e}")
        run_dir_to_clean = result.get("_run_dir")
        if run_dir_to_clean:
            shutil.rmtree(run_dir_to_clean, ignore_errors=True)
            logger.info(f"Cleaned up output dir: {run_dir_to_clean}")


# ─── FastAPI app ──────────────────────────────────────────────────────────────

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Video Analyzer v2")
templates = Jinja2Templates(directory="templates")
app.mount("/v2/results", StaticFiles(directory=OUTPUT_DIR), name="v2_results")


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("analyzer.html", {"request": request})


@app.get("/v2/status/{task_id}")
async def task_status(task_id: str):
    with _tasks_lock:
        task = _tasks.get(task_id)
    if task is None:
        return JSONResponse({"status": "not_found"}, status_code=404)
    if task["status"] == "processing":
        return JSONResponse({"status": "processing", "stage": task.get("stage", "")})
    return JSONResponse(task["result"])


@app.post("/v2/analyze-upload")
async def analyze_upload(
    files: List[UploadFile] = File(...),
    language: str = Form("English"),
    style: str = Form("standard"),
):
    if not files:
        return JSONResponse({"status": "error", "reason": "no files"}, status_code=400)

    Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
    run_ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(OUTPUT_DIR) / f"run_{run_ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    f = files[0]
    dest = Path(UPLOAD_DIR) / f.filename
    with open(dest, "wb") as out:
        shutil.copyfileobj(f.file, out)
    base_name = dest.stem
    video_out = run_dir / base_name
    video_out.mkdir(exist_ok=True)

    task_id = str(uuid.uuid4())
    with _tasks_lock:
        _tasks[task_id] = {"status": "processing", "stage": "starting", "result": None}

    t = threading.Thread(
        target=_run_task,
        args=(task_id, process_video_v2, str(dest), str(video_out), base_name, language, style),
        daemon=True,
    )
    t.start()
    return JSONResponse({"status": "processing", "task_id": task_id})


@app.post("/v2/analyze-url")
async def analyze_url(
    url: str = Form(...),
    language: str = Form("English"),
    style: str = Form("standard"),
):
    run_ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_tmpl = str(Path(UPLOAD_DIR) / f"ytdl_{run_ts}.%(ext)s")
    try:
        r = subprocess.run(
            ["yt-dlp", url, "-o", out_tmpl, "--no-playlist",
             "-f", "best[ext=mp4]/best", "--merge-output-format", "mp4"],
            capture_output=True, text=True, timeout=600,
        )
        if r.returncode != 0:
            return JSONResponse({"status": "error", "reason": f"yt-dlp: {r.stderr[:300]}"}, status_code=400)
    except FileNotFoundError:
        return JSONResponse({"status": "error", "reason": "yt-dlp not installed"}, status_code=500)
    except subprocess.TimeoutExpired:
        return JSONResponse({"status": "error", "reason": "download timeout"}, status_code=500)

    downloaded = list(Path(UPLOAD_DIR).glob(f"ytdl_{run_ts}.*"))
    if not downloaded:
        return JSONResponse({"status": "error", "reason": "file not found after download"}, status_code=500)

    video_path = str(downloaded[0])
    base_name  = Path(video_path).stem
    run_dir    = Path(OUTPUT_DIR) / f"run_{run_ts}"
    video_out  = run_dir / base_name
    video_out.mkdir(parents=True, exist_ok=True)

    task_id = str(uuid.uuid4())
    with _tasks_lock:
        _tasks[task_id] = {"status": "processing", "stage": "starting", "result": None}

    t = threading.Thread(
        target=_run_task,
        args=(task_id, process_video_v2, video_path, str(video_out), base_name, language, style),
        daemon=True,
    )
    t.start()
    return JSONResponse({"status": "processing", "task_id": task_id})


# ─── JSON API ─────────────────────────────────────────────────────────────────

LANG_MAP = {
    "en": "English", "de": "German", "fr": "French", "es": "Spanish",
    "it": "Italian", "pt": "Portuguese", "ru": "Russian", "ja": "Japanese",
    "zh": "Chinese", "ko": "Korean", "nl": "Dutch", "pl": "Polish",
    "ar": "Arabic",  "tr": "Turkish",  "cs": "Czech",   "sv": "Swedish",
}


class AnalyzeRequest(BaseModel):
    video_url: str
    languages: List[str] = ["en"]
    style: str = "standard"
    client_reference_id: str = ""
    webhook_url: str = ""
    # ignored fields kept for compat
    tag_count: int = 10
    secondary_tag_count: int = 7
    category_count: int = 10


def _check_api_key(x_api_key: Optional[str]):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")


_CHROME_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)


def _download_video(video_url: str, run_ts: str) -> str:
    """Download video via yt-dlp (supports mp4, HLS, most platforms). Returns local path."""
    out_tmpl = str(Path(UPLOAD_DIR) / f"api_{run_ts}.%(ext)s")
    r = subprocess.run(
        ["yt-dlp", video_url, "-o", out_tmpl, "--no-playlist",
         "-f", "best[ext=mp4]/best", "--merge-output-format", "mp4",
         "--user-agent", _CHROME_UA,
         "--add-header", "Accept-Language:en-US,en;q=0.9"],
        capture_output=True, text=True, timeout=600,
    )
    if r.returncode != 0:
        raise RuntimeError(f"yt-dlp error: {r.stderr[:400]}")
    downloaded = list(Path(UPLOAD_DIR).glob(f"api_{run_ts}.*"))
    if not downloaded:
        raise RuntimeError("File not found after download")
    return str(downloaded[0])


def _api_task(task_id: str, req: AnalyzeRequest):
    """Full pipeline: download → process → return result."""
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{task_id[:8]}"
    Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)

    # Download
    with _tasks_lock:
        _tasks[task_id]["stage"] = "downloading"
    try:
        video_path = _download_video(req.video_url, run_ts)
    except Exception as e:
        return {"status": "error", "reason": str(e),
                "client_reference_id": req.client_reference_id}

    langs     = req.languages or ["en"]
    lang_code = langs[0].lower()
    language  = LANG_MAP.get(lang_code, "English")
    extra     = langs[1:]  # remaining codes for extra SEO passes

    base_name = Path(video_path).stem
    run_dir   = Path(OUTPUT_DIR) / f"api_{run_ts}"
    video_out = run_dir / base_name
    video_out.mkdir(parents=True, exist_ok=True)

    with _tasks_lock:
        _tasks[task_id]["stage"] = "analyzing"

    result = process_video_v2(
        video_path, str(video_out), base_name, language, req.style,
        extra_languages=extra,
        tag_count=req.tag_count,
        secondary_tag_count=req.secondary_tag_count,
        category_count=req.category_count,
    )
    if req.client_reference_id:
        result["client_reference_id"] = req.client_reference_id

    result["_run_dir"] = str(run_dir)

    try:
        os.remove(video_path)
        logger.info(f"Deleted video: {video_path}")
    except Exception as e:
        logger.warning(f"Could not delete video {video_path}: {e}")

    return result


@app.post("/api/v2/analyze")
async def api_analyze(
    body: AnalyzeRequest,
    x_api_key: Optional[str] = Header(default=None),
):
    _check_api_key(x_api_key)
    task_id = str(uuid.uuid4())
    with _tasks_lock:
        _tasks[task_id] = {"status": "processing", "stage": "queued", "result": None}

    _task_queue.put((task_id, _api_task, (task_id, body), {}, body.webhook_url))
    queue_pos = _task_queue.qsize()
    return JSONResponse({"status": "processing", "task_id": task_id,
                         "client_reference_id": body.client_reference_id,
                         "queue_position": queue_pos})


@app.get("/api/v2/status/{task_id}")
async def api_task_status(
    task_id: str,
    x_api_key: Optional[str] = Header(default=None),
):
    _check_api_key(x_api_key)
    with _tasks_lock:
        task = _tasks.get(task_id)
    if task is None:
        return JSONResponse({"status": "not_found"}, status_code=404)
    if task["status"] == "processing":
        return JSONResponse({"status": "processing", "stage": task.get("stage", ""),
                             "queue_pending": _task_queue.qsize()})
    return JSONResponse(task["result"])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("video_processor_v4:app", host="0.0.0.0", port=8000, log_level="info", workers=1)

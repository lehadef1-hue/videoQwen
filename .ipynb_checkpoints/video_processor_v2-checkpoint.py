"""
video_processor_v2.py — Advanced adult video analyzer with SEO output.
Runs on :8001 by default.
"""

import os, sys, cv2, base64, io, json, re, logging, shutil, subprocess, uuid, threading
from datetime import datetime
from pathlib import Path
from glob import glob
from typing import List, Dict, Optional, Tuple

import requests
from PIL import Image
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

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
  "Double Penetration" = two simultaneous penetrations visible at same time
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

GROUPS:
  "Solo" = one performer alone throughout
  "Threesome" = exactly 3 performers
  "Group" = 4+ performers together
  "Gangbang" = 3+ on 1 — ALSO add "Group"
  "Lesbian" = female-female only, no males present at all
  "Bisexual Male" = male performs sex with both male AND female
  "Femdom" = female dominant, male visibly submissive

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


def build_seo_prompt(description: str, categories: List[str], orientation: str, language: str) -> str:
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

3. PRIMARY TAGS — exactly 5 long-tail keyword phrases (3–6 words each).
   - Exact search queries real users type. Based on acts, appearance, setting.
   - Format: lowercase phrases, e.g. "amateur outdoor sex video"

4. SECONDARY TAGS — exactly 7 shorter keyword phrases (2–4 words).
   - Broader supporting search terms. Mix acts, appearance, category keywords.
   - Format: lowercase phrases

5. SEO DESCRIPTION — 2–3 short paragraphs (80–120 words total).
   - Natural readable prose, not keyword stuffing. Start with main action and performers.
   - Mention key visual details, setting, mood. Tasteful language — no explicit profanity.
   - Keyword-rich but reads naturally.

--- OUTPUT ---
Return ONLY valid JSON, no markdown. All text in {language}.
{{"meta_title":"...","meta_description":"...","primary_tags":[...],"secondary_tags":[...],"seo_description":"..."}}"""


FRAME_PROMPT = """You receive {frame_count} frames (indexed 0 to {last_idx}) from a video.

TASK: Score and select the 5 best frames for display quality.

SCORING — start at 10, subtract:
  -5 if ANY performer has closed eyes
  -4 if image is blurry (motion blur, out-of-focus)
  -3 if any performer's face is not fully visible or cut off
  -2 if image is dark or overexposed
  -1 if no explicit sexual act is clearly visible

SELECTION:
  - Evaluate all frames
  - Pick exactly 5 with HIGHEST scores (or all if fewer than 5 available)
  - For each selected frame: return index (0-based), score (1-10), reason (5–15 words)

--- OUTPUT ---
Return ONLY valid JSON, no markdown, no extra text:
{{"frames":[{{"index":3,"score":8,"reason":"..."}},...]}}"""


FINAL_FRAME_PROMPT = """You receive {frame_count} frames (indexed 0 to {last_idx}).
These are already pre-selected as the BEST candidates from the full video.

TASK: Pick the final 5 frames for display + 1 thumbnail.

SELECTION RULES:
  - Choose exactly 5 frames (or fewer if less than 5 available)
  - VARIETY is the top priority: different acts, positions, angles, moments
  - Still reject: closed eyes (-5), blurry (-4), face cut off (-3), dark/overexposed (-2)
  - Re-score each chosen frame 1–10
  - For each: return index (0-based), score (1-10), reason (5–15 words)

THUMBNAIL (1 frame from your 5):
  - Best overall quality: sharp, eye contact, explicit action clearly visible

--- OUTPUT ---
Return ONLY valid JSON, no markdown, no extra text:
{{"frames":[{{"index":3,"score":9,"reason":"..."}},...],  "thumbnailIndex":3}}"""


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
        cats = [c for c in cats if c.lower() not in {"double penetration", "bisexual male"}]

    if "gangbang" in cats_lower and "group" not in cats_lower:
        cats = [c for c in cats if c.lower() != "gangbang"]

    if "solo" in cats_lower:
        cats = [c for c in cats if c.lower() not in {
            "group", "threesome", "gangbang", "double penetration", "bisexual male", "lesbian"
        }]

    if orientation == "gay" or "lesbian" in {c.lower() for c in cats}:
        cats = [c for c in cats if c.lower() != "femdom"]

    return cats


def _parse_frame_candidates(
    parsed: Optional[Dict],
    frames: List[Image.Image],
    timestamps: Optional[List[Optional[float]]] = None,
) -> List[Dict]:
    if not parsed:
        return []
    result = []
    for item in parsed.get("frames", []):
        idx = item.get("index")
        if not isinstance(idx, int) or idx < 0 or idx >= len(frames):
            continue
        c: Dict = {
            "frame": frames[idx],
            "score": int(item.get("score", 5)),
            "reason": str(item.get("reason", ""))[:80],
            "ts": None,
            "ts_fmt": "",
        }
        if timestamps and idx < len(timestamps) and timestamps[idx] is not None:
            c["ts"] = float(round(timestamps[idx], 1))
            c["ts_fmt"] = _fmt_ts(timestamps[idx])
        result.append(c)
    return result


# ─── Core processor ───────────────────────────────────────────────────────────

def process_video_v2(
    video_path: str,
    output_dir: str,
    base_name: str,
    language: str = "English",
    style: str = "standard",
) -> Dict:
    logger.info(f"Processing v2: {base_name} | lang={language} style={style}")
    try:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        mid   = total_frames // 2
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
            {"temperature": 0.45, "top_p": 0.85, "max_tokens": 1400},
        )
        p1 = extract_json_from_response(raw1)
        if not p1:
            return {"status": "error", "reason": "pass1 invalid response"}

        description = p1.get("description", "").strip()
        orientation = p1.get("orientation", "straight")
        if orientation not in VALID_ORIENTATIONS:
            orientation = "straight"
        cats_raw   = _normalize_cats(p1.get("categories", []))[:15]
        watermarks = [str(w).strip() for w in (p1.get("watermarks") or []) if str(w).strip()]

        # Convert key scene frame indices → timestamps
        key_scenes = []
        for ks in (p1.get("key_scenes") or []):
            fi = ks.get("frame")
            if isinstance(fi, int) and 0 <= fi < len(ts_1a):
                key_scenes.append({
                    "ts": float(round(ts_1a[fi], 1)),
                    "formatted": _fmt_ts(ts_1a[fi]),
                    "desc": str(ks.get("desc", "")).strip(),
                })

        final_categories = validate_categories(cats_raw, orientation)
        logger.info(f"Pass 1: orient={orientation} cats={len(final_categories)} watermarks={watermarks} scenes={len(key_scenes)}")

        # ── Pass 2a: Frame scoring first half ───────────────────────────────────
        frames_2a, ts_2a = extract_key_frames_ts(video_path, 25, start_at=skip8, end_at=mid)
        raw2a = call_vision_model(
            FRAME_PROMPT.format(frame_count=len(frames_2a), last_idx=len(frames_2a) - 1),
            frames_2a,
            {"temperature": 0.3, "top_p": 0.80, "max_tokens": 1200},
        )
        candidates_a = _parse_frame_candidates(extract_json_from_response(raw2a), frames_2a, ts_2a)
        logger.info(f"Pass 2a: candidates={len(candidates_a)}")

        # ── Pass 2b: Frame scoring second half ──────────────────────────────────
        frames_2b, ts_2b = extract_key_frames_ts(video_path, 25, start_at=mid, end_at=end92)
        raw2b = call_vision_model(
            FRAME_PROMPT.format(frame_count=len(frames_2b), last_idx=len(frames_2b) - 1),
            frames_2b,
            {"temperature": 0.3, "top_p": 0.80, "max_tokens": 1200},
        )
        candidates_b = _parse_frame_candidates(extract_json_from_response(raw2b), frames_2b, ts_2b)
        logger.info(f"Pass 2b: candidates={len(candidates_b)}")

        # ── Merge → Pass 3: Final selection ────────────────────────────────────
        all_cands = sorted(candidates_a + candidates_b, key=lambda x: x["score"], reverse=True)
        top10 = all_cands[:10]
        top5  = top10[:5]
        thumb_frame = top10[0]["frame"] if top10 else None

        if top10:
            frames_3 = [c["frame"] for c in top10]
            ts_3     = [c.get("ts") for c in top10]
            raw3 = call_vision_model(
                FINAL_FRAME_PROMPT.format(frame_count=len(frames_3), last_idx=len(frames_3) - 1),
                frames_3,
                {"temperature": 0.3, "top_p": 0.80, "max_tokens": 1200},
            )
            p3 = extract_json_from_response(raw3)
            cands_3 = _parse_frame_candidates(p3, frames_3, ts_3)
            thumb_idx_3 = (p3 or {}).get("thumbnailIndex")
            if cands_3:
                top5 = sorted(cands_3, key=lambda x: x["score"], reverse=True)[:5]
                if isinstance(thumb_idx_3, int) and 0 <= thumb_idx_3 < len(frames_3):
                    thumb_frame = frames_3[thumb_idx_3]
                elif top5:
                    thumb_frame = top5[0]["frame"]
            logger.info(f"Pass 3: candidates={len(cands_3)}")

        # ── Pass SEO ────────────────────────────────────────────────────────────
        seo_ref = [thumb_frame] if thumb_frame else frames_1a[:1]
        raw_seo = call_vision_model(
            build_seo_prompt(description, final_categories, orientation, language),
            seo_ref,
            {"temperature": 0.3, "top_p": 0.85, "max_tokens": 900},
        )
        p_seo = extract_json_from_response(raw_seo) or {}
        meta_title      = p_seo.get("meta_title", "").strip()
        meta_desc       = p_seo.get("meta_description", "").strip()
        primary_tags    = [t.strip() for t in p_seo.get("primary_tags", []) if t.strip()][:5]
        secondary_tags  = [t.strip() for t in p_seo.get("secondary_tags", []) if t.strip()][:7]
        seo_description = p_seo.get("seo_description", "").strip()
        logger.info(f"Pass SEO: title={len(meta_title)} desc={len(meta_desc)} tags={len(primary_tags)+len(secondary_tags)}")

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
        if thumb_frame is not None:
            thumb_path = os.path.join(output_dir, f"{base_name}_thumb.jpg")
            thumb_frame.save(thumb_path, quality=88, optimize=True)

        meta = {
            "description": description,
            "categories": final_categories[:15],
            "orientation": orientation,
            "watermarks": watermarks,
            "performers": performers,
            "key_scenes": key_scenes,
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
            "meta_title": meta_title,
            "meta_description": meta_desc,
            "primary_tags": primary_tags,
            "secondary_tags": secondary_tags,
            "seo_description": seo_description,
            "thumbnail": _to_url(thumb_path),
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


# ─── Task store ───────────────────────────────────────────────────────────────
# {task_id: {"status": "processing"|"done"|"error", "stage": str, "result": dict}}
_tasks: Dict[str, Dict] = {}
_tasks_lock = threading.Lock()


def _run_task(task_id: str, fn, *args, **kwargs):
    """Run fn(*args, **kwargs) in a thread; store result in _tasks."""
    try:
        result = fn(*args, **kwargs)
    except Exception as e:
        result = {"status": "error", "reason": str(e)}
    with _tasks_lock:
        _tasks[task_id]["status"] = result.get("status", "error")
        _tasks[task_id]["result"] = result


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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("video_processor_v2:app", host="0.0.0.0", port=8001, log_level="info", workers=1)

import os
import sys
import cv2
import base64
import io
import json
import re
import logging
from datetime import datetime
from pathlib import Path
from glob import glob
from PIL import Image
import requests
from typing import List, Dict, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Опциональный модуль распознавания исполнителей (требует insightface)
try:
    from performer_finder import identify_performers as _identify_performers
    PERFORMER_RECOGNITION_AVAILABLE = True
except ImportError:
    PERFORMER_RECOGNITION_AVAILABLE = False

# ─── Логирование ───────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s | %(levelname)-7s | %(threadName)-10s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("video_analyzer")

# ─── Конфигурация ──────────────────────────────────────────────────────────────
MODEL_SERVER_URL = os.getenv("MODEL_SERVER_URL", "http://localhost:8080/generate")

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

# ─── Промпт 1: анализ контента (теги + описание) ──────────────────────────────
ANALYSIS_PROMPT = """You receive {frame_count} key frames sampled evenly across the full video.

TASK: Analyze the video content and return structured metadata.

--- ORIENTATION ---
Choose EXACTLY ONE value: straight | gay | shemale
- straight  = male+female sex, OR all-female (lesbian = category tag, NOT an orientation)
- gay       = ONLY male performers having sex with each other — no women present at all
- shemale   = trans woman (MTF): visibly female body (breasts, feminine shape) WITH a penis visible

CRITICAL: "lesbian" is NOT a valid orientation — use "straight" instead.
CRITICAL: if you see a penis on a femininely-built performer → shemale, not straight.

--- DESCRIPTION ---
Write a vivid, explicit, dirty description (3-5 sentences). 
Use raw, vulgar slang and dirty talk style. Describe the performers, their bodies, positions, actions in graphic detail. 
Write like a horny human would describe the scene to a friend — be crude, playful and nasty. Do NOT be clinical or polite. 
NEVER start with "This video", "In this video", "The video" or similar — jump straight into describing the action.

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

WHEN IN DOUBT → OMIT. Fewer correct tags > many hallucinated tags.
Return AT MOST 10 categories total.

--- STUDIO / WATERMARK ---
Look for any STATIC text overlay, watermark, or logo visible across the frames.
This is typically: a website URL (e.g. "brazzers.com"), a studio brand name (e.g. "Brazzers", "Reality Kings"),
or a channel name. It usually appears in a corner (top-left, bottom-right, etc.) and is semi-transparent or white text.

Rules:
- Return the text EXACTLY as it appears (preserve capitalisation and dots)
- If it's a URL like "www.site.com" or "site.com" — return just the domain, e.g. "site.com"
- If you see a logo/brand without readable text, describe it briefly, e.g. "Brazzers logo"
- Return null if no watermark is clearly readable in at least 2 frames

--- OUTPUT ---
Return ONLY valid JSON, no markdown, no extra text:
{{"orientation":"straight","description":"...","studio":null,"categories":[...]}}"""

# --- DESCRIPTION ---
# Write exactly 2–3 sentences (max 120 words total) using raw explicit slang. Start directly with the action.
# NEVER start with "This video", "In this scene", "The video", etc.
# Be concise — do NOT list every detail, focus only on the main act.


# ─── Промпт 2: выбор лучших кадров (с оценкой) ────────────────────────────────
FRAME_PROMPT = """You receive {frame_count} frames (indexed 0 to {last_idx}) from a video.

TASK: Score and select the best frames for display.

SCORING CRITERIA — assign a score 1–10 to each frame you consider:
  Start at 10, subtract:
  -5 if ANY performer has closed eyes (immediate disqualifier, max score = 5)
  -3 if any performer's face is not fully visible or cut off
  -2 if no performer makes direct eye contact with the camera
  -1 if no explicit sexual act is clearly visible
  -4 if image is blurry (motion blur, out-of-focus) — strong disqualifier, max score = 6
  -2 if image is dark or overexposed

SELECTION:
  - Evaluate all frames
  - Pick exactly 5 with HIGHEST scores (must have score ≥ 4)
  - If fewer than 5 frames score ≥ 4, take the best available
  - Prefer VARIETY: different positions and moments, avoid consecutive indices
  - For each selected frame: return index (0-based), score (1-10), reason (5–15 words)

THUMBNAIL (1 frame from your 5):
  - Pick the frame with the HIGHEST score
  - On tie: prefer eye contact + action visible

--- OUTPUT ---
Return ONLY valid JSON, no markdown, no extra text:
{{"frames":[{{"index":3,"score":8,"reason":"..."}},...],  "thumbnailIndex":3}}"""
# ─── Вспомогательные функции ───────────────────────────────────────────────────

VALID_ORIENTATIONS = {"straight", "gay", "shemale"}

def pil_to_base64(img: Image.Image) -> str:
    buffered = io.BytesIO()
    img.convert("RGB").save(buffered, format="JPEG", quality=82, optimize=True)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def call_vision_model(
    prompt: str,
    images: List[Image.Image],
    sampling: Optional[Dict] = None
) -> str:
    base64_images = [pil_to_base64(img) for img in images]
    payload = {
        "prompt": prompt,
        "base64_images": base64_images,
        "sampling_params": sampling or {
            "temperature": 0.65,
            "top_p": 0.90,
            "max_tokens": 1200
        }
    }
    try:
        logger.debug(f"POST → {MODEL_SERVER_URL} | images: {len(images)}")
        resp = requests.post(MODEL_SERVER_URL, json=payload, timeout=420)
        resp.raise_for_status()
        data = resp.json()
        output = data.get("output", "").strip()
        logger.debug(f"Ответ модели: {len(output)} символов")
        return output
    # except Exception as e:
    except Exception:
        logger.exception("Ошибка вызова модели")
        return ""


def extract_json_from_response(text: str) -> Optional[Dict]:
    if not text:
        return None

    def _try_parse(s: str) -> Optional[Dict]:
        s = re.sub(r',\s*(?=[}\]])', '', s).strip()
        # Fix: model puts "thumbnailIndex": N inside frames array instead of as sibling key
        # {"frames": [..., "thumbnailIndex": 3}  →  {"frames": [...], "thumbnailIndex": 3}
        s = re.sub(r'(?<!\]),\s*"thumbnailIndex":\s*(\d+)\s*\}\s*$', r'], "thumbnailIndex": \1}', s)
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            return None

    # 1. Markdown code blocks: ```json ... ``` or ``` ... ```
    for pattern in (r'```json\s*([\s\S]*?)```', r'```\s*([\s\S]*?)```'):
        m = re.search(pattern, text)
        if m:
            result = _try_parse(m.group(1))
            if result is not None:
                return result

    # 2. Find all top-level {...} blocks using a bracket counter (avoids greedy over-match)
    candidates = []
    depth = 0
    start = None
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

    # Try longest candidates first (most complete JSON)
    for candidate in sorted(candidates, key=len, reverse=True):
        result = _try_parse(candidate)
        if result is not None:
            return result

    # 3. Fallback: extract fields via regex from truncated JSON
    result: Dict = {}
    for key in ("orientation", "description", "studio"):
        m = re.search(rf'"{key}"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
        if m:
            result[key] = m.group(1)
        else:
            m2 = re.search(rf'"{key}"\s*:\s*(null)', text)
            if m2:
                result[key] = None
    cats_m = re.search(r'"categories"\s*:\s*\[([^\]]*)', text)
    if cats_m:
        result["categories"] = re.findall(r'"((?:[^"\\]|\\.)*)"', cats_m.group(1))
    if result:
        logger.warning("Used regex fallback to recover partial JSON")
        return result

    logger.error("Failed to extract valid JSON from model response")
    logger.debug(f"Проблемный текст:\n{text[:800]}")
    return None


def validate_categories(categories: List[str], orientation: Optional[str]) -> List[str]:
    """Удаляет теги, противоречащие ориентации и логически несовместимые."""
    cats = list(categories)
    cats_lower = {c.lower() for c in cats}

    # Теги, требующие пениса — запрещены для лесби/gay-female видео
    requires_penis = {"titty fucking", "handjob", "cumshot", "facial", "creampie", "big dick"}
    # Теги, требующие вагину/женского тела
    requires_vagina = {"pussy licking", "squirt", "creampie"}

    if orientation == "gay":
        # Gay (только мужчины): нет вагины, нет женских тегов, нет страпона
        forbidden = requires_vagina | {
            "lesbian", "milf", "squirt", "pregnant", "lactating",
            "pussy licking", "pegging", "strap on",
        }
        cats = [c for c in cats if c.lower() not in forbidden]

    elif orientation == "shemale":
        # Shemale (транс-женщина с пенисом):
        # - Lesbian = только женщины, пениса нет → неверно
        # - Gay = только мужчины → неверно
        # - Pegging = проникновение страпоном → у шимейл реальный пенис, не страпон
        # - Pregnant / Lactating → биологически невозможно для MTF
        forbidden = {"lesbian", "gay", "pegging", "strap on", "pregnant", "lactating"}
        cats = [c for c in cats if c.lower() not in forbidden]

    elif orientation == "straight":
        # Straight: не может быть gay-тегов
        cats = [c for c in cats if c.lower() != "gay"]
        # Если присутствует Lesbian — убрать теги требующие пениса
        if "lesbian" in cats_lower:
            cats = [c for c in cats if c.lower() not in requires_penis]

    cats_lower = {c.lower() for c in cats}

    # Групповые правила
    has_group_like = any(x in cats_lower for x in ["group", "threesome", "gangbang"])
    if not has_group_like:
        # Без группы: DP и bisexual male невозможны
        cats = [c for c in cats if c.lower() not in {"double penetration", "bisexual male"}]

    # Gangbang требует group
    if "gangbang" in cats_lower and "group" not in cats_lower:
        cats = [c for c in cats if c.lower() != "gangbang"]

    # Solo несовместимо с группой
    if "solo" in cats_lower:
        cats = [c for c in cats if c.lower() not in {"group", "threesome", "gangbang", "double penetration", "bisexual male", "lesbian"}]

    # Femdom требует мужчины — не применимо для gay/lesbian
    if orientation == "gay" or ("lesbian" in {c.lower() for c in cats}):
        cats = [c for c in cats if c.lower() != "femdom"]

    # # Убрать "HD" — это не тег контента
    # cats = [c for c in cats if c.lower() != "hd"]

    return cats


def extract_key_frames(
    video_path: str,
    target_count: int = 25,
    start_at: Optional[int] = None,
    end_at: Optional[int] = None,
) -> List[Image.Image]:
    """Извлекает target_count кадров равномерно из диапазона [start_at, end_at].

    По умолчанию пропускает первые и последние 4% (интро/аутро).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 8:
        cap.release()
        raise ValueError("Video too short")

    if start_at is None:
        start_at = max(1, int(total_frames * 0.04))
    if end_at is None:
        end_at = total_frames - max(1, int(total_frames * 0.04))

    usable = max(1, end_at - start_at)
    step = max(1, usable // target_count)

    frames = []
    for i in range(start_at, end_at, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(rgb))
        if len(frames) >= target_count:
            break
    cap.release()
    logger.info(f"Extracted {len(frames)} frames from {video_path} (pos {start_at}–{end_at} / {total_frames})")
    return frames


def extract_frames_for_selection(video_path: str, target_count: int = 25) -> List[Image.Image]:
    """Извлекает кадры для выбора thumbnail — из центральной части (8%–92%), плотнее."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if total_frames < 8:
        raise ValueError("Video too short")
    skip = max(1, int(total_frames * 0.08))
    return extract_key_frames(video_path, target_count=target_count, start_at=skip, end_at=total_frames - skip)


def _normalize_cats(raw: list) -> List[str]:
    """Нормализует список категорий: проверяет по ALLOWED_CATEGORIES_LOWER, дедуплицирует."""
    result, seen = [], set()
    for cat in raw:
        key = cat.lower().strip()
        if key in ALLOWED_CATEGORIES_LOWER and key not in seen:
            result.append(ALLOWED_CATEGORIES_LOWER[key])
            seen.add(key)
    return result


def _parse_frame_candidates(parsed: Optional[Dict], frames: List[Image.Image]) -> List[Dict]:
    """Извлекает кандидатов кадров из ответа модели в единый формат."""
    if not parsed:
        return []
    candidates = []
    for item in parsed.get("frames", []):
        idx = item.get("index")
        if not isinstance(idx, int) or idx < 0 or idx >= len(frames):
            continue
        candidates.append({
            "frame": frames[idx],
            "score": int(item.get("score", 5)),
            "reason": str(item.get("reason", ""))[:80],
        })
    return candidates


def process_video(video_path: str, output_dir: str, base_name: str) -> Dict:
    """Четырёхпроходный анализ (2 API-вызова на теги + 2 на выбор кадров).

    Pass 1a — первые 25 кадров (4%–50%)  → orientation + description + categories_A
    Pass 1b — вторые 25 кадров (50%–96%) → categories_B (дополнение)
    Pass 2a — первые 25 кадров контента (8%–50%)  → scored frame candidates_A
    Pass 2b — вторые 25 кадров контента (50%–92%) → scored frame candidates_B
    Итог: categories_A ∪ categories_B; top-5 кадров по скору из candidates_A+B.
    """
    logger.info(f"Processing: {base_name}")
    try:
        _cap = cv2.VideoCapture(video_path)
        total_frames_count = int(_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        _cap.release()
        mid = total_frames_count // 2
        skip4  = max(1, int(total_frames_count * 0.04))
        skip8  = max(1, int(total_frames_count * 0.08))
        end92  = total_frames_count - skip8

        # ── Pass 1a: теги + описание (полное видео 4%–92%) ─────────────────────
        frames_1a = extract_key_frames(video_path, 25, start_at=skip4, end_at=end92)
        if len(frames_1a) < 4:
            return {"status": "skipped", "reason": "too few frames"}

        raw1a = call_vision_model(
            ANALYSIS_PROMPT.format(frame_count=len(frames_1a)),
            frames_1a,
            {"temperature": 0.45, "top_p": 0.85, "max_tokens": 1000}
        )
        p1a = extract_json_from_response(raw1a)
        if not p1a:
            return {"status": "error", "reason": "pass1a invalid response"}

        description  = p1a.get("description", "").strip()
        orientation  = p1a.get("orientation", "straight")
        if orientation not in VALID_ORIENTATIONS:
            orientation = "straight"
        cats_a       = _normalize_cats(p1a.get("categories", []))[:15]
        studio_raw   = p1a.get("studio")
        studio       = str(studio_raw).strip() if studio_raw and str(studio_raw).strip().lower() not in ("null", "none", "") else None
        logger.info(f"Pass 1a: orient={orientation} cats={len(cats_a)} studio={studio!r}")

        final_categories = validate_categories(cats_a, orientation)

        # ── Pass 2a: выбор кадров (первая половина контента) ────────────────────
        frames_2a = extract_key_frames(video_path, 25, start_at=skip8, end_at=mid)
        raw2a = call_vision_model(
            FRAME_PROMPT.format(frame_count=len(frames_2a), last_idx=len(frames_2a) - 1),
            frames_2a,
            {"temperature": 0.3, "top_p": 0.80, "max_tokens": 1200}
        )
        p2a        = extract_json_from_response(raw2a)
        candidates_a = _parse_frame_candidates(p2a, frames_2a)
        thumb_idx_a  = (p2a or {}).get("thumbnailIndex")
        logger.info(f"Pass 2a: candidates={len(candidates_a)}")

        # ── Pass 2b: выбор кадров (вторая половина контента) ────────────────────
        frames_2b = extract_key_frames(video_path, 25, start_at=mid, end_at=end92)
        raw2b = call_vision_model(
            FRAME_PROMPT.format(frame_count=len(frames_2b), last_idx=len(frames_2b) - 1),
            frames_2b,
            {"temperature": 0.3, "top_p": 0.80, "max_tokens": 1200}
        )
        p2b = extract_json_from_response(raw2b)
        candidates_b = _parse_frame_candidates(p2b, frames_2b)
        thumb_idx_b  = (p2b or {}).get("thumbnailIndex")
        logger.info(f"Pass 2b: candidates={len(candidates_b)}")

        # ── Мерж кандидатов: сортируем по скору, берём top-5 ────────────────────
        all_candidates = sorted(
            candidates_a + candidates_b,
            key=lambda x: x["score"],
            reverse=True
        )
        top5 = all_candidates[:5]

        # Thumbnail: кандидат с максимальным скором
        # (если модель явно указала thumbnail в одном из проходов — берём его если скор высокий)
        thumb_frame = None
        explicit_thumb = None
        if isinstance(thumb_idx_a, int) and 0 <= thumb_idx_a < len(frames_2a):
            explicit_thumb = frames_2a[thumb_idx_a]
        elif isinstance(thumb_idx_b, int) and 0 <= thumb_idx_b < len(frames_2b):
            explicit_thumb = frames_2b[thumb_idx_b]

        thumb_frame = explicit_thumb if explicit_thumb is not None else (top5[0]["frame"] if top5 else None)

        # # ── Распознавание исполнителей (опционально) ─────────────────────────────
        # performers: List[str] = []
        # if PERFORMER_RECOGNITION_AVAILABLE:
        #     # Передаём кадры из обоих наборов для максимального покрытия лиц
        #     performers = _identify_performers(frames_1a)
        #     if performers:
        #         logger.info(f"Performers identified: {performers}")
        
        # ── Распознавание исполнителей (опционально) ─────────────────────────────
        performers: List[str] = []
        if PERFORMER_RECOGNITION_AVAILABLE:
            frames_face = extract_key_frames(video_path, 100, start_at=skip4, end_at=end92)
            performers = _identify_performers(frames_face)
            if performers:
                logger.info(f"Performers identified: {performers}")

        # ── Сохранение ──────────────────────────────────────────────────────────
        os.makedirs(output_dir, exist_ok=True)
        saved_frames = []
        for i, cand in enumerate(top5):
            path = os.path.join(output_dir, f"{base_name}_frame_{i:03d}.jpg")
            cand["frame"].save(path, quality=85, optimize=True)
            saved_frames.append({"score": cand["score"], "path": path, "reason": cand["reason"]})

        thumb_path = None
        if thumb_frame is not None:
            thumb_path = os.path.join(output_dir, f"{base_name}_thumb.jpg")
            thumb_frame.save(thumb_path, quality=88, optimize=True)

        meta = {
            "description": description,
            "categories": final_categories[:15],
            "orientation": orientation,
            "studio": studio,
            "performers": performers,
        }
        with open(os.path.join(output_dir, f"{base_name}_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        logger.info(f"Done: {base_name} | orient={orientation} studio={studio!r} cats={len(final_categories)} frames={len(saved_frames)}")
        return {
            "status": "ok",
            "thumbnail": thumb_path,
            "top5_frames": saved_frames,
            "categories": final_categories,
            "orientation": orientation,
            "studio": studio,
            "performers": performers,
            "description": description[:400] if description else None,
        }

    except Exception as e:
        logger.exception(f"Critical error processing {video_path}")
        return {"status": "error", "reason": str(e)}


# ─── FastAPI приложение ────────────────────────────────────────────────────────
app = FastAPI(
    title="Adult Video Analyzer",
    description="Анализ видео: thumbnails, категории, описание",
    version="0.1"
)

class ProcessRequest(BaseModel):
    input_dir: str = "/workspace/video/videos"
    output_dir: str = "/workspace/video/result"


# Поддержка Jinja2 и статических файлов
templates = Jinja2Templates(directory="templates")
app.mount("/results", StaticFiles(directory="/workspace/video/result"), name="results")


@app.get("/browse")
async def browse_results(request: Request):
    base = Path("/workspace/video/result")
    runs = []

    for run_dir in sorted(base.iterdir(), reverse=True):
        if not run_dir.is_dir() or not run_dir.name.startswith("run_"):
            continue
        run_time = run_dir.name.replace("run_", "").replace("_", " ")
        videos = []

        for video_dir in run_dir.iterdir():
            if not video_dir.is_dir():
                continue
            meta_file = video_dir / f"{video_dir.name}_meta.json"
            if not meta_file.exists():
                continue

            with open(meta_file, encoding="utf-8") as f:
                meta = json.load(f)

            thumb_rel = f"/results/{run_dir.name}/{video_dir.name}/{video_dir.name}_thumb.jpg"
            thumb_abs = Path("/workspace/video/result") / run_dir.name / video_dir.name / f"{video_dir.name}_thumb.jpg"
            thumb = thumb_rel if thumb_abs.exists() else None

            frames = []
            for f in sorted(video_dir.glob(f"{video_dir.name}_frame_*.jpg")):
                rel_path = f"/results/{run_dir.name}/{video_dir.name}/{f.name}"
                frames.append(rel_path)

            videos.append({
                "name": video_dir.name,
                "thumbnail": thumb,
                "frames": frames[:8],
                "categories": meta.get("categories", []),
                "orientation": meta.get("orientation", ""),
                "studio": meta.get("studio") or "",
                "performers": meta.get("performers") or [],
                "description": meta.get("description", "—")
            })

        runs.append({
            "name": run_dir.name,
            "time": run_time,
            "videos": videos
        })

    return templates.TemplateResponse(
        "index.html",
        {"request": request, "runs": runs}
    )


@app.post("/process")
def process_videos_endpoint(req: ProcessRequest):
    logger.info(f"Request → input: {req.input_dir} | output: {req.output_dir}")

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir_name = f"run_{run_timestamp}"
    base_output = Path(req.output_dir)
    this_run_dir = base_output / run_dir_name
    this_run_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Результаты этого запуска → {this_run_dir}")

    videos = []
    for pattern in ("*.mp4", "*.mkv", "*.webm", "*.mov"):
        videos.extend(glob(os.path.join(req.input_dir, pattern)))

    if not videos:
        return {"status": "nothing_to_process", "videos_found": 0}

    results = []
    for path in sorted(videos):
        base_name = Path(path).stem
        video_output_dir = this_run_dir / base_name
        video_output_dir.mkdir(exist_ok=True)

        result = process_video(path, str(video_output_dir), base_name)
        results.append({
            "file": base_name,
            "result": result,
            "output_folder": str(video_output_dir)
        })

    return {
        "status": "processed",
        "run_folder": str(this_run_dir),
        "run_timestamp": run_timestamp,
        "results": results
    }


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting video analyzer server on :8000")
    uvicorn.run(
        "video_processor:app",           # ← поменяй на имя своего файла без .py
        host="0.0.0.0",
        port=8000,
        log_level="info",
        workers=1
    )
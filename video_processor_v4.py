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
    '3d Animation', '69 Pose', 'Amateur', 'American', 'Anal', 'Anime', 'Arab', 'Asian',
    'Ass To Mouth', 'Babysitter', 'Ballbusting', 'BBW', 'BDSM', 'Bathrooms',
    'Beach', 'Behind The Scenes', 'Big Ass', 'Big Dick', 'Big Tits', 'Bikini',
    'Bisexual Male', 'Blonde', 'Blowjob', 'Bondage', 'Boots', 'Brazilian',
    'British', 'Brunette', 'Bukkake', 'CBT', 'CFNM', 'Cartoon', 'Casting',
    'Celebrity', 'Cheerleader', 'Chinese', 'Chubby', 'Clamps', 'Close Up',
    'College', 'Compilation', 'Cosplay', 'Couple', 'Cowgirl', 'Creampie',
    'Crossdresser', 'Cuckold', 'Cum In Mouth', 'Cumshot', 'Czech',
    'Deep Throat', 'Device Bondage', 'Diaper', 'Doggystyle',
    'Double Anal Penetration', 'Double Penetration', 'Double Pussy Penetration',
    'Ebony', 'Electricity Play', 'Enema', 'European', 'Exhibitionism',
    'Face Sitting', 'Facial', 'Fantasy Character', 'Female Orgasm', 'Femdom',
    'Fetish', 'Fingering', 'Fishnet', 'Fisting', 'Foot Fetish', 'Footjob',
    'French', 'Fuck Machine', 'Funny', 'Furry', 'Futanari', 'Gagging', 'Gags',
    'Game Character', 'Gameplay Video', 'Gangbang', 'Gaping', 'German',
    'Glory Hole', 'Goth', 'Granny', 'Group', 'Gym', 'HD', 'Hairy',
    'Handcuffs', 'Handjob', 'Hanging Up', 'Hardcore', 'Hentai', 'High Heels',
    'Hotel', 'Humiliation', 'Indian', 'Insect Fetish', 'Interracial', 'Italian',
    'Japanese Censored', 'Japanese Uncensored', 'Korean', 'Lactating', 'Latex',
    'Latina', 'Leather', 'Leggings', 'Lesbian', 'Lingerie', 'Long Legs', 'MILF',
    'Maid Uniform', 'Mask', 'Massage', 'Masturbation', 'Mature', 'Medical',
    'Midget', 'Military', 'Muscular Man', 'Natural Tits', 'Nipples', 'Nudist',
    'Nurse Uniform', 'Office', 'Oiled', 'Old/Young', 'OnlyFans', 'Outdoor',
    'POV', 'Party', 'Pegging', 'Piercing', 'Pissing', 'Pornstar', 'Pregnant',
    'Public', 'Pumping toy', 'Pussy Licking', 'Red Head', 'Rimjob', 'Russian',
    'School Uniform', 'Secretary', 'Sensor Deprivation', 'Shaved',
    'Shibari Bondage', 'Shower', 'Skinny', 'Slave', 'Slave Cage', 'Small Cock',
    'Small Tits', 'Smoking', 'Snuffing', 'Soap Play', 'Solo', 'Spanish',
    'Spandex', 'Spanking', 'Sport', 'Squirt', 'Step Fantasy', 'Stockings',
    'Strap On', 'Striptease', 'Swallow Cum', 'Swimsuit', 'Swingers', 'Tattoo',
    'Teacher', 'Teen', 'Tentacle', 'Thai', 'Threesome', 'Tight Clothes',
    'Tickling', 'Titty Fucking', 'Toilet', 'Toys', 'Transport Fetish', 'Uniform',
    'Vertical Video', 'Vintage', 'Virtual Reality', 'Voyeur', 'Wax', 'Webcam',
    'Whipping', 'Wife Sharing', 'Wrestling', 'Yoga',
]
ALLOWED_CATEGORIES_LOWER = {c.lower(): c for c in ALLOWED_CATEGORIES}
VALID_ORIENTATIONS = {"straight", "gay", "shemale"}

# Aliases: common model-generated names → canonical ALLOWED_CATEGORIES name
_CAT_ALIASES: Dict[str, str] = {
    # Femdom synonyms
    "female dominance": "Femdom",
    "female domination": "Femdom",
    "dominatrix": "Femdom",
    "dominant woman": "Femdom",
    "dominant female": "Femdom",
    "female dom": "Femdom",
    "submissive man": "Femdom",
    "male submission": "Femdom",
    "male submissive": "Femdom",
    "power exchange": "Femdom",
    "sexual role reversal": "Femdom",
    "role reversal": "Femdom",
    "femdom bdsm": "Femdom",
    # BDSM synonyms
    "bondage and discipline": "BDSM",
    "sadomasochism": "BDSM",
    "s&m": "BDSM",
    "sm": "BDSM",
    "d/s": "BDSM",
    "ds": "BDSM",
    # "discipline" убрано — "self-discipline", "military discipline" дали бы BDSM
    "dom/sub": "BDSM",
    "domination": "BDSM",
    # "submission" убрано — слишком широкое слово
    "kinky": "BDSM",
    "kink": "BDSM",
    # Group/gangbang synonyms
    "multiple partners": "Group",
    "multiple people": "Group",
    "orgy": "Group",
    "multi partner": "Group",
    # Sex acts
    # "penetrative sex" убрано — нет прямой категории, Amateur здесь неверно
    "oral sex": "Blowjob",
    "blow job": "Blowjob",
    "blowjob": "Blowjob",
    "cunnilingus": "Pussy Licking",
    "hand job": "Handjob",
    "titfuck": "Titty Fucking",
    "tit fuck": "Titty Fucking",
    "butt plug": "Toys",
    "dildo": "Toys",
    "vibrator": "Toys",
    # "toy" убрано — слишком короткое, цепляется через n-gram в составных словах
    "sex toy": "Toys",
    "sex toys": "Toys",
    "pumping": "Pumping toy",
    "pump toy": "Pumping toy",
    # Appearance
    "curvy": "BBW",
    "plus size": "BBW",
    "large woman": "BBW",
    "heavy set": "BBW",
    "fat": "BBW",
    "redhead": "Red Head",
    "red hair": "Red Head",
    "auburn": "Red Head",
    "dark hair": "Brunette",
    "black hair": "Brunette",
    "inked": "Tattoo",
    "tattooed": "Tattoo",
    # Settings
    "outdoors": "Outdoor",
    "outside": "Outdoor",
    "nature": "Outdoor",
    "public place": "Public",
    "car": "Transport Fetish",
    "vehicle": "Transport Fetish",
    # Styles
    "pov sex": "POV",
    "point of view": "POV",
    "first person": "POV",
    "high definition": "HD",
    "1080p": "HD",
    "4k": "HD",
    "homemade": "Amateur",
    "home video": "Amateur",
    "selfie": "OnlyFans",
    "creator content": "OnlyFans",
    # Ethnicity
    # "black" убрано — слишком широкое ("BlackLatex" → Ebony было бы ложным)
    "african": "Ebony",
    "african american": "Ebony",
    "east asian": "Asian",
    "southeast asian": "Asian",
    "chinese": "Chinese",
    "korean": "Korean",
    "japanese": "Asian",    # ambiguous — can't determine censored/uncensored from word alone
    "vietnamese": "Asian",
    "latina": "Latina",
    "latin": "Latina",
    "middle eastern": "Arab",
    # Age
    "young": "Teen",
    "18 year old": "Teen",
    "19 year old": "Teen",
    "young adult": "Teen",
    "older woman": "Mature",
    "older man": "Mature",
    "cougar": "MILF",
    "age gap": "Old/Young",
    "older younger": "Old/Young",
    # Bondage/BDSM sub-categories
    "restrained": "Bondage",
    "tied up": "Bondage",
    "tied": "Bondage",
    "bound": "Bondage",
    "rope bondage": "Bondage",
    "ropes": "Bondage",
    "chains": "Bondage",
    "cuffs": "Handcuffs",
    "metal cuffs": "Handcuffs",
    "ball gag": "Gags",
    "gagged": "Gags",
    "muzzle": "Gags",
    "taped mouth": "Gags",
    "mouth tape": "Gags",
    "blindfold": "Sensor Deprivation",
    "blindfolded": "Sensor Deprivation",
    "suspension bondage": "Hanging Up",
    "suspended": "Hanging Up",
    "nipple clamps": "Clamps",
    "clothespins": "Clamps",
    "flogging": "Whipping",
    "flogger": "Whipping",
    "whipped": "Whipping",
    "cropped": "Whipping",
    # "crop" убрано — "CropTop" (одежда) давал бы ложный Whipping
    "riding crop": "Whipping",
    "paddle": "Spanking",
    "paddled": "Spanking",
    "spanked": "Spanking",
    "butt slapping": "Spanking",
    "electric wand": "Electricity Play",
    "e-stim": "Electricity Play",
    "hot wax": "Wax",
    "candle wax": "Wax",
    "wax dripping": "Wax",
    "cage": "Slave Cage",
    "japanese rope": "Shibari Bondage",
    "shibari": "Shibari Bondage",
    "kinbaku": "Shibari Bondage",
    "spreader bar": "Device Bondage",
    "stocks": "Device Bondage",
    "rack": "Device Bondage",
    # Femdom/dominance
    "female led": "Femdom",
    "mistress": "Femdom",
    "goddess": "Femdom",
    "dominance": "Femdom",
    # "dominant" убрано — слишком широкое (dominant partner в обычном сексе → BDSM было бы ложным)
    "slave": "BDSM",
    "submissive": "BDSM",
    "sub": "BDSM",
    "sub/dom": "BDSM",
    "pegged": "Pegging",
    "strap-on": "Strap On",
    "strapon": "Strap On",
    "strap on dildo": "Strap On",
    # Sex acts extra
    "bj": "Blowjob",
    "fellatio": "Blowjob",
    "deep throat bj": "Deep Throat",
    "throat fucking": "Deep Throat",
    "face fucking": "Deep Throat",
    "mouth fucking": "Deep Throat",
    # "vaginal sex"/"intercourse"/"fucking"/"sex" убраны — Amateur это качество съёмки,
    # не акт; лучше дропнуть через _normalize_cats чем добавить неверную категорию
    "licking pussy": "Pussy Licking",
    "eating out": "Pussy Licking",
    "eating pussy": "Pussy Licking",
    "oral": "Pussy Licking",
    "analingus": "Rimjob",
    "ass licking": "Rimjob",
    "rim job": "Rimjob",
    "rimming": "Rimjob",
    "dp": "Double Penetration",
    "double pen": "Double Penetration",
    "fisted": "Fisting",
    "fist fuck": "Fisting",
    "squirting": "Squirt",
    "female ejaculation": "Squirt",
    "cum on face": "Facial",
    "cum shot": "Cumshot",
    "cumming": "Cumshot",
    "ejaculation": "Cumshot",
    "internal cum": "Creampie",
    "internal ejaculation": "Creampie",
    "creampied": "Creampie",
    "massage sex": "Massage",
    "rubdown": "Massage",
    "oiled massage": "Massage",
    # "pump" убрано — слишком широкое (pump iron = качалка, water pump и т.д.)
    "vacuum pump": "Pumping toy",
    "penis pump": "Pumping toy",
    "tickled": "Tickling",
    "feather": "Tickling",
    "golden shower": "Pissing",
    "pee": "Pissing",
    "urination": "Pissing",
    "watersports": "Pissing",
    "smoking fetish": "Smoking",
    "cigarette": "Smoking",
    # Appearance extra
    "chubby girl": "Chubby",
    "overweight": "Chubby",
    "big boobs": "Big Tits",
    "large breasts": "Big Tits",
    "huge tits": "Big Tits",
    "busty": "Big Tits",
    "big booty": "Big Ass",
    "large ass": "Big Ass",
    "huge ass": "Big Ass",
    "big butt": "Big Ass",
    "well-endowed": "Big Dick",
    "huge cock": "Big Dick",
    "large cock": "Big Dick",
    "large penis": "Big Dick",
    "shiny skin": "Oiled",
    "oily": "Oiled",
    "lubricant": "Oiled",
    "lube": "Oiled",
    "hairy pussy": "Hairy",
    "unshaved": "Hairy",
    "natural bush": "Hairy",
    "pubic hair": "Hairy",
    "body hair": "Hairy",
    "ink": "Tattoo",
    "tattoos": "Tattoo",
    "dwarf": "Midget",
    "dwarfism": "Midget",
    "short stature": "Midget",
    "pregnant belly": "Pregnant",
    "baby bump": "Pregnant",
    "breast milk": "Lactating",
    "lactation": "Lactating",
    "milk spraying": "Lactating",
    "crossdressing": "Crossdresser",
    "cross-dresser": "Crossdresser",
    "cross dresser": "Crossdresser",
    "trap": "Crossdresser",
    "futanari": "Futanari",
    "futa": "Futanari",
    # Performer count extra
    "solo girl": "Solo",
    "solo male": "Solo",
    "masturbation": "Masturbation",
    "girl on girl": "Lesbian",
    "female on female": "Lesbian",
    "ffm": "Threesome",
    "mmf": "Threesome",
    "three way": "Threesome",
    "threeway": "Threesome",
    "3some": "Threesome",
    "group sex": "Group",
    "four way": "Group",
    "five way": "Group",
    "foursome": "Group",
    "gang bang": "Gangbang",
    "gang-bang": "Gangbang",
    "interracial sex": "Interracial",
    "mixed race": "Interracial",
    "bi male": "Bisexual Male",
    "bisexual": "Bisexual Male",
    "cuckold porn": "Cuckold",
    "cuckolding": "Cuckold",
    "cuck": "Cuckold",
    "hotwife": "Cuckold",
    # Settings extra
    "outside sex": "Outdoor",
    "outdoor sex": "Outdoor",
    "forest": "Outdoor",
    "park": "Outdoor",
    "beach sex": "Beach",
    "ocean": "Beach",
    "poolside": "Outdoor",
    "in public": "Public",
    "public sex": "Public",
    "exhibition": "Public",
    "exhibitionism": "Exhibitionism",
    "office sex": "Office",
    "workplace": "Office",
    "at the gym": "Gym",
    "fitness": "Gym",
    "shower sex": "Shower",
    "bathroom sex": "Bathrooms",
    "bathtub": "Bathrooms",
    "toilet sex": "Toilet",
    "restroom": "Toilet",
    "in a car": "Transport Fetish",
    "on a train": "Transport Fetish",
    "airplane": "Transport Fetish",
    "sports": "Sport",
    "athletic": "Sport",
    "yoga mat": "Yoga",
    "yoga pose": "Yoga",
    # Camera/format extra
    "vertical": "Vertical Video",
    "portrait mode": "Vertical Video",
    "portrait video": "Vertical Video",
    "phone video": "Vertical Video",
    "vr porn": "Virtual Reality",
    "360 video": "Virtual Reality",
    "fisheye": "Virtual Reality",
    "4k video": "HD",
    "high quality": "HD",
    "old footage": "Vintage",
    "vhs": "Vintage",
    "retro": "Vintage",
    "voyeuristic": "Voyeur",
    "hidden camera": "Voyeur",
    "spy cam": "Voyeur",
    "peeping": "Voyeur",
    "webcam show": "Webcam",
    "live stream": "Webcam",
    "only fans": "OnlyFans",
    "onlyfans content": "OnlyFans",
    "creator": "OnlyFans",
    "professional production": "Pornstar",
    "studio production": "Pornstar",
    "porn actress": "Pornstar",
    "porn actor": "Pornstar",
    "pornographic actress": "Pornstar",
    "casting couch": "Casting",
    "audition": "Casting",
    # Costumes/roleplay extra
    "cosplay costume": "Cosplay",
    "anime cosplay": "Cosplay",
    "schoolgirl": "School Uniform",
    "school girl": "School Uniform",
    "maid": "Maid Uniform",
    "maid outfit": "Maid Uniform",
    "nurse": "Nurse Uniform",
    "nurse outfit": "Nurse Uniform",
    "knee high boots": "Boots",
    "thigh high boots": "Boots",
    "leather boots": "Boots",
    "bra and panties": "Lingerie",
    "underwear": "Lingerie",
    "sexy lingerie": "Lingerie",
    "stockings and heels": "Stockings",
    "thigh highs": "Stockings",
    "nylons": "Stockings",
    "latex suit": "Latex",
    "pvc": "Latex",
    "rubber": "Latex",
    "elf": "Fantasy Character",
    "demon": "Fantasy Character",
    "vampire": "Fantasy Character",
    "fairy": "Fantasy Character",
    "fantasy": "Fantasy Character",
    "foot worship": "Foot Fetish",
    "feet worship": "Foot Fetish",
    "foot licking": "Foot Fetish",
    "toe sucking": "Foot Fetish",
    # Animation extra
    "2d animation": "Anime",
    "animated": "Anime",
    "hentai anime": "Hentai",
    "3d cgi": "3d Animation",
    "cgi animation": "3d Animation",
    "western cartoon": "Cartoon",
    "animated show": "Cartoon",
    "gameplay": "Gameplay Video",
    "video game footage": "Gameplay Video",
    "gaming video": "Gameplay Video",
    "anthropomorphic": "Furry",
    "fursuit": "Furry",
    "tentacles": "Tentacle",
    # Ethnicity extra
    "persian": "Arab",
    "iranian": "Arab",
    "turkish": "Arab",
    "moroccan": "Arab",
    "north african": "Arab",
    "south asian": "Indian",
    "desi": "Indian",
    "bangladeshi": "Indian",
    "pakistani": "Indian",
    "sri lankan": "Indian",
    "thai girl": "Thai",
    "thailand": "Thai",
    "japanese girl": "Japanese Uncensored",
    "uncensored japanese": "Japanese Uncensored",
    "censored japanese": "Japanese Censored",
    "mosaic censored": "Japanese Censored",
    "pixel censored": "Japanese Censored",
    "ebony woman": "Ebony",
    "black woman": "Ebony",
    "black man": "Ebony",
    # Age extra
    "barely legal": "Teen",
    "18yo": "Teen",
    "19yo": "Teen",
    "young woman": "Teen",
    "milf sex": "MILF",
    "sexy mom": "MILF",
    "hot mom": "MILF",
    "stepmom": "Step Fantasy",
    "stepdaughter": "Step Fantasy",
    "step daughter": "Step Fantasy",
    "stepson": "Step Fantasy",
    "grandma": "Granny",
    "grandmother": "Granny",
    "elderly woman": "Granny",
    "mature woman": "Mature",
    "mature man": "Mature",
    "middle aged": "Mature",
    "older couple": "Mature",
    # Other fetish
    "fetish video": "Fetish",
    "unusual fetish": "Fetish",
    "insect": "Insect Fetish",
    "bugs": "Insect Fetish",
    "diaper fetish": "Diaper",
    "adult diapers": "Diaper",
    "abdl": "Diaper",
    "snuff fantasy": "Snuffing",
    "execution roleplay": "Snuffing",
    "death roleplay": "Snuffing",
    # "soap" убрано — "SoapOpera" (турецкие/иранские сериалы) дало бы ложный "Soap Play"
    "lather": "Soap Play",
    "soap lather": "Soap Play",
    "bubbly": "Soap Play",
    "sensory deprivation": "Sensor Deprivation",
    "hood": "Sensor Deprivation",
    # CamelCase варианты которые генерирует модель
    "brune": "Brunette",
    "brunete": "Brunette",
    "middle eastern": "Arab",
    "middleeastern": "Arab",
    "homevideo": "Amateur",     # CamelCase вариант
    "captivity": "Bondage",
    "prisoner": "Bondage",
    "prisoner scenario": "Bondage",
    "solitary confinement": "Bondage",
    "captive": "Bondage",
    "kidnap": "Bondage",
    "kidnapping": "Bondage",
    "abduction": "Bondage",
    # "distress" убрано — эмоциональный дистресс ≠ BDSM (турецкие сериалы)
    # "indoor", "household", "home" убраны — слишком широкие, неверное мэппинг в Amateur
    "homemade video": "Amateur",
    # ── New-category aliases ─────────────────────────────────────────────────────
    # Blowjob
    "sucking dick": "Blowjob",
    "cock sucking": "Blowjob",
    "sucking cock": "Blowjob",
    "dick sucking": "Blowjob",
    "mouth sex": "Blowjob",
    "suck": "Blowjob",
    # Ass To Mouth
    "atm": "Ass To Mouth",
    "ass 2 mouth": "Ass To Mouth",
    "anus to mouth": "Ass To Mouth",
    # Gagging (oral gagging during sex — not a device)
    "gag on cock": "Gagging",
    "choking on cock": "Gagging",
    "gagging on dick": "Gagging",
    "throat gagging": "Gagging",
    # Face Sitting
    "facesitting": "Face Sitting",
    "face sit": "Face Sitting",
    "queening": "Face Sitting",
    "sit on face": "Face Sitting",
    "sitting on face": "Face Sitting",
    "face riding": "Face Sitting",
    # Fingering
    "finger fucking": "Fingering",
    "fingered": "Fingering",
    "finger fuck": "Fingering",
    "finger insertion": "Fingering",
    # Gaping
    "gape": "Gaping",
    "gaping ass": "Gaping",
    "gaping hole": "Gaping",
    "gaping pussy": "Gaping",
    "gaping anus": "Gaping",
    "anal gape": "Gaping",
    # Cum In Mouth
    "cum in mouth": "Cum In Mouth",
    "mouth cumshot": "Cum In Mouth",
    "cum mouth": "Cum In Mouth",
    # Swallow Cum
    "swallowing cum": "Swallow Cum",
    "swallows cum": "Swallow Cum",
    "cum swallow": "Swallow Cum",
    "cum swallowing": "Swallow Cum",
    "swallow": "Swallow Cum",
    # Double Anal Penetration
    "dap": "Double Anal Penetration",
    "double anal": "Double Anal Penetration",
    "double anal sex": "Double Anal Penetration",
    # Double Pussy Penetration
    "dvp": "Double Pussy Penetration",
    "double vaginal": "Double Pussy Penetration",
    "double vaginal penetration": "Double Pussy Penetration",
    "double pussy": "Double Pussy Penetration",
    # Fuck Machine
    "sex machine": "Fuck Machine",
    "fucking machine": "Fuck Machine",
    "mechanical dildo": "Fuck Machine",
    "fucking device": "Fuck Machine",
    "auto fuck": "Fuck Machine",
    "fucking machine sex": "Fuck Machine",
    # Exhibitionism
    "flashing": "Exhibitionism",
    "public nudity": "Exhibitionism",
    "public exposure": "Exhibitionism",
    "streaking": "Exhibitionism",
    "exhibitionist": "Exhibitionism",
    "indecent exposure": "Exhibitionism",
    # Humiliation
    "humiliated": "Humiliation",
    "degradation": "Humiliation",
    "degraded": "Humiliation",
    "verbal humiliation": "Humiliation",
    "verbal abuse": "Humiliation",
    "degrade": "Humiliation",
    "spitting on": "Humiliation",
    # Ballbusting
    "ball busting": "Ballbusting",
    "ball bust": "Ballbusting",
    "ball kicking": "Ballbusting",
    "genital kicking": "Ballbusting",
    "kick in balls": "Ballbusting",
    "nut busting": "Ballbusting",
    # CBT
    "cock and ball torture": "CBT",
    "cock torture": "CBT",
    "ball torture": "CBT",
    "genital torture": "CBT",
    "cock ball torture": "CBT",
    # Enema
    "enema play": "Enema",
    "anal douche": "Enema",
    "rectal enema": "Enema",
    "water enema": "Enema",
    # Medical
    "medical fetish": "Medical",
    "clinical play": "Medical",
    "doctor fetish": "Medical",
    "gyno exam": "Medical",
    "doctor patient": "Medical",
    "medical exam": "Medical",
    "latex gloves fetish": "Medical",
    # CFNM
    "clothed female naked male": "CFNM",
    "clothed females naked males": "CFNM",
    "cfnm fetish": "CFNM",
    # Masturbation (extra synonyms)
    "jerking off": "Masturbation",
    "jerk off": "Masturbation",
    "wanking": "Masturbation",
    "wank": "Masturbation",
    "solo masturbation": "Masturbation",
    "self pleasure": "Masturbation",
    "touching herself": "Masturbation",
    "self stimulation": "Masturbation",
    # 69 Pose
    "sixty nine": "69 Pose",
    "sixty-nine": "69 Pose",
    "69 position": "69 Pose",
    "mutual oral": "69 Pose",
    "soixante-neuf": "69 Pose",
    "69 sex": "69 Pose",
    # Step Fantasy
    "stepbro": "Step Fantasy",
    "stepsis": "Step Fantasy",
    "step brother": "Step Fantasy",
    "step sister": "Step Fantasy",
    "step mom": "Step Fantasy",
    "step dad": "Step Fantasy",
    "step son": "Step Fantasy",
    "stepson": "Step Fantasy",
    "stepfather": "Step Fantasy",
    "stepmother": "Step Fantasy",
    "stepfamily": "Step Fantasy",
    "step family": "Step Fantasy",
    "step roleplay": "Step Fantasy",
    "taboo family": "Step Fantasy",
    "family roleplay": "Step Fantasy",
    # Behind The Scenes
    "bts": "Behind The Scenes",
    "making of": "Behind The Scenes",
    "behind scenes": "Behind The Scenes",
    "making-of": "Behind The Scenes",
    "camera crew": "Behind The Scenes",
    # Compilation
    "best of": "Compilation",
    "multi clip": "Compilation",
    "various clips": "Compilation",
    "mixed clips": "Compilation",
    "compilation video": "Compilation",
    "clip compilation": "Compilation",
    # Nudist
    "naturist": "Nudist",
    "nudism": "Nudist",
    "naturism": "Nudist",
    "nude resort": "Nudist",
    # Cheerleader
    "cheerleader outfit": "Cheerleader",
    "cheerleader costume": "Cheerleader",
    "cheer uniform": "Cheerleader",
    # Secretary
    "secretary roleplay": "Secretary",
    "office secretary": "Secretary",
    "boss secretary": "Secretary",
    # Teacher
    "teacher roleplay": "Teacher",
    "teacher student": "Teacher",
    "professor": "Teacher",
    "classroom sex": "Teacher",
    "teacher fetish": "Teacher",
    # Babysitter
    "babysitter roleplay": "Babysitter",
    "babysitter scenario": "Babysitter",
    "babysitter sex": "Babysitter",
    # Military
    "army": "Military",
    "soldier": "Military",
    "military uniform": "Military",
    "camouflage": "Military",
    "army uniform": "Military",
    # Uniform (generic, not covered by specifics)
    "police uniform": "Uniform",
    "police officer": "Uniform",
    "firefighter": "Uniform",
    "flight attendant": "Uniform",
    "cop uniform": "Uniform",
    # Wrestling
    "catfight": "Wrestling",
    "sexfight": "Wrestling",
    "sex fight": "Wrestling",
    "wrestling match": "Wrestling",
    "erotic wrestling": "Wrestling",
    # Wife Sharing
    "shared wife": "Wife Sharing",
    "sharing wife": "Wife Sharing",
    "share the wife": "Wife Sharing",
    "hotwife sharing": "Wife Sharing",
    # Swingers
    "wife swap": "Swingers",
    "wife swapping": "Swingers",
    "couple swap": "Swingers",
    "couple swapping": "Swingers",
    "lifestyle couples": "Swingers",
    # Bikini
    "bikini sex": "Bikini",
    "in bikini": "Bikini",
    "bikini girl": "Bikini",
    "bikini babe": "Bikini",
    # Swimsuit
    "bathing suit": "Swimsuit",
    "one piece": "Swimsuit",
    "one-piece": "Swimsuit",
    "bathing costume": "Swimsuit",
    # Spandex
    "lycra": "Spandex",
    "spandex outfit": "Spandex",
    "spandex suit": "Spandex",
    # Leggings
    "yoga pants": "Leggings",
    "tight leggings": "Leggings",
    # Tight Clothes
    "skintight": "Tight Clothes",
    "tight outfit": "Tight Clothes",
    "tight dress": "Tight Clothes",
    "body con": "Tight Clothes",
    "bodycon": "Tight Clothes",
    # Mask
    "wearing mask": "Mask",
    "masked": "Mask",
    "anonymous mask": "Mask",
    "bdsm mask": "Mask",
    "gimp mask": "Mask",
    # Goth
    "gothic": "Goth",
    "emo": "Goth",
    "dark aesthetic": "Goth",
    "alternative girl": "Goth",
    # Piercing
    "body piercings": "Piercing",
    "nipple piercing": "Piercing",
    "genital piercing": "Piercing",
    "tongue piercing": "Piercing",
    "pierced": "Piercing",
    # Party
    "party sex": "Party",
    "club sex": "Party",
    "nightclub": "Party",
    "dorm party": "Party",
    "sex party": "Party",
    # Porn star alternate spelling
    "porn star": "Pornstar",
    # Hotwife as Cuckold variant
    "hotwife": "Cuckold",
    # ── Missing aliases from bdsmx.tube comparison ──────────────────────────────
    "leggins": "Leggings",          # опечатка в API bdsmx.tube
    "shaving": "Shaved",
    "shaves": "Shaved",
    "big natural tits": "Natural Tits",
    "big naturals": "Natural Tits",
    "large natural tits": "Natural Tits",
    "candid": "Voyeur",
    "candid camera": "Voyeur",
    "pantyhose": "Stockings",
    "tights": "Stockings",
    "hanging": "Hanging Up",
    "pumping device": "Pumping toy",
    "submissive women": "Femdom",
    "submissive woman": "Femdom",
    # New categories
    "doggystyle sex": "Doggystyle",
    "doggy style": "Doggystyle",
    "doggy": "Doggystyle",
    "doggy position": "Doggystyle",
    "rear entry": "Doggystyle",
    "cowgirl position": "Cowgirl",
    "girl on top": "Cowgirl",
    "woman on top": "Cowgirl",
    "reverse cowgirl": "Cowgirl",
    "riding position": "Cowgirl",
    "extreme close up": "Close Up",
    "close-up": "Close Up",
    "macro shot": "Close Up",
    "celebrity tape": "Celebrity",
    "sex tape": "Celebrity",
    "famous person": "Celebrity",
    "slave roleplay": "Slave",
    "sex slave": "Slave",
    "slave training": "Slave",
    "american porn": "American",
    "usa": "American",
    "us production": "American",
    "funny video": "Funny",
    "bloopers": "Funny",
    "comedy": "Funny",
    "muscular": "Muscular Man",
    "muscle man": "Muscular Man",
    "bodybuilder": "Muscular Man",
    "ripped": "Muscular Man",
    "long legs": "Long Legs",
    "legs": "Long Legs",
    "leggy": "Long Legs",
}

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
        "Write a vivid, explicit, dirty description (3–4 sentences, MAX 350 characters). "
        "Use raw, vulgar slang and dirty talk style. Describe performers, bodies, positions, actions in graphic detail. "
        "Write like a horny human — be crude, playful, nasty. Do NOT be clinical or polite. "
        "NEVER start with 'This video', 'In this video', or similar — jump straight into describing the action. "
        "Stop after 4 sentences maximum."
    ),
    "clean": (
        "Write a tasteful description (3–4 sentences, MAX 350 characters) suitable for mainstream platforms. "
        "Focus on mood, setting, performers' appearance, and the nature of the encounter. "
        "Avoid explicit sexual terms. NEVER start with 'This video', 'In this video', or similar. "
        "Stop after 4 sentences maximum."
    ),
    "cinematic": (
        "Write a cinematic description (3–4 sentences, MAX 350 characters) in the style of a film critic. "
        "Focus on visual composition, lighting quality, performers' chemistry, camera angles. "
        "Treat it as a review of visual and performative qualities. NEVER start with 'This video' or similar. "
        "Stop after 4 sentences maximum."
    ),
}

# ─── Prompts ──────────────────────────────────────────────────────────────────

def build_analysis_prompt(frame_count: int, ts_map: str, desc_style: str, language: str) -> str:
    return f"""You receive {frame_count} key frames sampled evenly across the full porn video.

FRAME TIMESTAMPS (seconds from video start):
{ts_map}

--- ORIENTATION ---
Choose EXACTLY ONE value: straight | gay | shemale
- straight  = male+female sex, OR all-female (lesbian = category tag, NOT an orientation)
- gay       = ONLY male performers having sex with each other — no women present at all
- shemale   = trans woman (MTF): visibly feminine body (breasts) WITH a penis visible

CRITICAL: "lesbian" is NOT a valid orientation → use "straight".
CRITICAL: feminine body + visible penis → "shemale", not "straight".

--- CONTENT TYPE ---
Before writing the description, identify what type of content this is:
• Real human performers only → describe normally
• Animated/CGI content (anime, 3D, cartoon, hentai) → say so explicitly in description
• MIXED: real human performer + animated/CGI elements (tentacles, monster, creature, animated penis/appendages) → CRITICAL: describe the non-human elements accurately. If penetration is by a tentacle, animated appendage, or CGI creature — call it "tentacle", "creature appendage", or "animated monster", NOT "penis" or "cock". Do NOT misidentify non-human objects as human anatomy.

--- DESCRIPTION ---
{desc_style}

--- WATERMARKS / ON-SCREEN TEXT ---
List ALL visible static text overlays, watermarks, logos across ALL frames.
Return exact text as array (e.g. ["brazzers.com", "pornhub.com"]). Return [] if none.

--- KEY SCENES ---
Identify 5–8 key moments or scene changes using the frame timestamps above.
Reference frames by their 0-based index. Write a NEUTRAL 1-sentence description (no explicit terms).

--- OUTPUT ---
Return ONLY valid JSON, no markdown. All text fields in {language}.
{{"orientation":"straight","description":"...","watermarks":["site.com"],"key_scenes":[{{"frame":2,"desc":"..."}}]}}"""


def build_categories_prompt(frame_count: int, orientation: str, description: str = "") -> str:
    """Dedicated categories-only prompt — model returns exact category names from ALLOWED_CATEGORIES."""
    cats_flat = ", ".join(f'"{c}"' for c in ALLOWED_CATEGORIES)
    desc_block = (
        f"\nVIDEO DESCRIPTION (context only — do NOT use to infer sexual act categories):\n{description}\n"
        if description else ""
    )
    return f"""You receive {frame_count} key frames from an adult video. Detected orientation: {orientation}
{desc_block}
TASK: Select 5–15 categories. HARD MAXIMUM: 15.

RULES FOR USING DESCRIPTION vs FRAMES:
  • Description is CONTEXT only — use it to confirm ethnicity, setting, performer appearance, performer count.
  • Sexual ACT categories (Anal, Pussy Licking, Deep Throat, Cumshot, Facial, Creampie, Fisting, Squirt, Handjob, Titty Fucking, Rimjob, Double Penetration, Pegging, Pissing) — ONLY from clear visual evidence in frames. NEVER infer from description text alone.
  • Appearance/size categories (Big Dick, Big Tits, Big Ass, BBW, Chubby) — ONLY from clear visual frames. Do NOT infer from description words like "thick", "huge", "big".

IMPORTANT: Many categories require NO explicit sexual acts — tag visual elements unconditionally:
  • Bondage/BDSM/Gags/Handcuffs = visible restraints or gag, regardless of sexual acts
  • Ethnicity (Arab, Asian, Ebony, Interracial, etc.) = performer appearance in frames or confirmed by description
  • Settings (Outdoor, Shower, Bathrooms, Office, etc.) = clearly visible location
  • Appearance (Blonde, Tattoo, etc.) = performer's visible features
  • Solo/Lesbian/Threesome/Group = performer count
You MUST always return a JSON array with at least 1 category. NEVER return an error or explanation.
CRITICAL: Do NOT tag categories based on watermarks, logos, or text visible on screen.

═══ PERFORMERS & APPEARANCE ═══
"Amateur"          = home/self-produced only: non-studio lighting + shaky/phone camera + ordinary domestic setting — ALL THREE required. DEFAULT is NOT amateur. Do NOT tag if studio lighting, smooth camera work, or porn studio watermark is visible.
"Pornstar"         = professional studio production with professional lighting, editing, professional performers. Do NOT tag together with "Amateur".
"Casting"          = STRICT: video must show an explicit casting/audition setup — a visible interviewer or off-camera voice conducting an interview/audition, and the performer being "recruited". Requires TWO people minimum. Do NOT tag for: solo performances, professional studio shoots without audition framing, amateur home videos, or any video where the casting scenario is not the central premise of the scene.
"OnlyFans"         = creator-style content: phone/selfie camera, direct address to camera/audience, casual home setting, solo or couple. Distinguished from Amateur by the direct-to-audience creator style. Do NOT tag if no direct camera address.
"Webcam"           = clearly live webcam recording: static wide-angle shot, solo performer facing camera directly, often grainy or low-resolution. Distinguished from OnlyFans by static framing and live-stream aesthetic.

"BBW"              = performer is clearly obese — very large belly, thick limbs, heavy overall build. Must be obvious at a glance. Do NOT tag both BBW and Chubby for the same performer.
"Chubby"           = performer is noticeably overweight but not obese — soft/rounded body, visible extra weight. Do NOT tag if performer is merely curvy or average build.
"Midget"           = performer with dwarfism / clearly very short stature
"Big Tits"         = breasts clearly larger than average — unmistakably large
"Big Ass"          = buttocks clearly larger than average — unmistakably large
"Big Dick"         = penis clearly larger than average — unmistakably large. ONLY from visual frame evidence, NOT from description text.
"Small Cock"       = penis clearly smaller than average — unmistakably small
"Big Tits"         = breasts clearly larger than average — unmistakably large
"Small Tits"       = breasts clearly smaller than average — flat or very small chest
"Natural Tits"     = breasts visibly natural/unaugmented — soft, natural shape with natural sag. Do NOT tag if breast implants are visible or suspected.
"Nipples"          = nipples are the clear visual focus — prominently featured, close-up shots
"Oiled"            = body clearly covered in oil/lube — skin visibly shiny/wet from oil
"Skinny"           = performer has clearly slim/thin body — noticeably underweight or very lean
"Shaved"           = performer's pubic area is completely shaved/bare — smooth skin, no hair visible. Requires a clear genital close-up where the shaved area is the visual focus.
"Piercing"         = performer has clearly visible body piercings (genital, nipple, tongue) prominently featured
"Goth"             = performer has clearly goth aesthetic — dark makeup, black clothing, dark/alternative style

"Blonde"           = performer(s) with clearly blonde hair
"Brunette"         = performer(s) with clearly dark brown or black hair
"Red Head"         = performer(s) with clearly red or auburn hair
"Hairy"            = dense, clearly visible natural pubic bush in a genital close-up — that is the ONLY trigger. Head/leg/arm/armpit hair = irrelevant. Shaved or trimmed pubic area → use "Shaved", never "Hairy".
"Tattoo"           = performer(s) have prominent, clearly visible tattoos (large or multiple). Do NOT tag for a single small/barely visible tattoo — tattoos should be a notable part of the performer's appearance.

"Teen"             = performer visibly appears 18-19 years old (young adult face/body)
"Mature"           = performer(s) clearly appear middle-aged (40s–50s) regardless of role. Age must be apparent from face/body — do NOT tag based on context alone.
"MILF"             = same age range as Mature (40s+) but specifically presented in a "sexy older woman" fantasy role — confident, experienced, often paired with younger partner. Tag BOTH "Mature" and "MILF" when both apply. Do NOT tag MILF for all older women — only when the "desirable older woman" fantasy angle is the clear focus.
"Granny"           = clearly elderly (60s+) female performer
"Old/Young"        = clear age gap between partners — one clearly much older, one clearly young adult

"Asian"            = performer(s) clearly of East or Southeast Asian ethnicity
"Japanese Censored"     = Japanese production with mosaic/pixel censorship over genitals
"Japanese Uncensored"   = Japanese production without any censorship
"Thai"             = performer(s) clearly Thai ethnicity or clearly Thai production
"Indian"           = performer(s) clearly South Asian / Indian subcontinent ethnicity
"Arab"             = performer(s) clearly Arab or Middle Eastern ethnicity
"Ebony"            = performer(s) clearly of Black/African ethnicity
"Interracial"      = partners of visibly different races actively having sex together
"Latina"           = performer(s) clearly of Latin American / Hispanic ethnicity
"Chinese"          = performer(s) clearly Chinese ethnicity or clearly Chinese production
"Korean"           = performer(s) clearly Korean ethnicity or clearly Korean production
"Russian"          = performer(s) clearly Russian or Eastern European ethnicity / Russian production
"Brazilian"        = performer(s) clearly Brazilian ethnicity or clearly Brazilian production
"Czech"            = clearly Czech production (Prague casting style, Czech studio look)
"French"           = performer(s) clearly French or clearly French production
"German"           = performer(s) clearly German or clearly German production
"Italian"          = performer(s) clearly Italian or clearly Italian production
"British"          = performer(s) clearly British or clearly UK production
"Spanish"          = performer(s) clearly Spanish or clearly Spanish production
"European"         = performer(s) clearly European but nationality not more specific

"Pregnant"         = performer has a clearly visible, significantly enlarged pregnant belly — third trimester size. Do NOT tag if belly is ambiguous, slightly rounded, or performer is simply overweight. Only tag when pregnancy is unmistakable.
"Lactating"        = breast milk is visibly dripping, spraying, or being expressed/sucked. Do NOT tag if breasts are merely large or wet — actual milk flow must be clearly visible on screen.
"Crossdresser"     = male performer dressed in women's clothing/lingerie as a fetish
"American"         = performer(s) clearly American or clearly American production style
"Celebrity"        = celebrity sex tape or content featuring a recognizable public figure (actor, musician, influencer, athlete)
"Muscular Man"     = male performer with clearly defined, prominent muscular physique — visibly large muscles, athletic/bodybuilder build
"Long Legs"        = female performer with notably long legs prominently featured — legs are the clear visual focus (close-ups, poses highlighting leg length)

═══ PERFORMER COUNT (ACTIVE participants only — not bystanders) ═══
"Solo"             = exactly ONE performer performing alone — zero partners in scene
"Lesbian"          = ONLY females having sex with each other — zero males at all
"Threesome"        = exactly 3 performers ALL actively participating together
"Group"            = 4 or more performers ALL actively participating together
"Gangbang"         = 3+ performers focusing on 1 — MUST also add "Group"
"Bisexual Male"    = one male ACTIVELY has sex with both a male AND a female in the same scene
"Cuckold"          = clearly staged cuckold scenario — one partner watches while other is with third
"Couple"           = exactly TWO performers — one-on-one sex between two people
"Swingers"         = two couples swapping partners or group sex as lifestyle swap scenario
"Wife Sharing"     = man sharing his partner with another man — clearly staged sharing scenario

═══ SEXUAL ACTS (tag only if clearly visible in multiple frames, NOT implied) ═══
"Anal"             = VERY STRICT. A penis can only be in ONE orifice at a time — if vaginal penetration is clearly visible anywhere in the video, do NOT also tag Anal unless "Double Penetration" is also present (two penetrations simultaneously). Required visual evidence: the penetrating object must be clearly seen entering/inside the anus in an unambiguous close-up — the anus must be identifiably visible as the insertion point, distinguishable from the vagina. FORBIDDEN triggers: doggy style, from-behind position, side-view thrusting, overhead/behind angle without clear insertion view, buttocks visible but orifice not identifiable. Rear-entry is vaginal by default. Must appear in multiple clear frames. ANY ambiguity → do NOT tag.
"Double Penetration"= TWO orifices penetrated SIMULTANEOUSLY — both visible at the same time
"Blowjob"          = oral sex performed on a penis — mouth on penis clearly visible. Tag for regular blowjob. Use "Deep Throat" instead ONLY when full shaft visibly disappears into throat with gagging. Always also tag "Blowjob" when "Deep Throat" is tagged.
"Fisting"          = entire fist inside vagina/anus — extremely rare, be very strict
"Fingering"        = fingers inserted into vagina or anus — must be clearly visible insertion, not just touching
"Masturbation"     = performer visibly masturbating (hand on own genitals, self-stimulation). Tag additionally "Solo" if alone.
"Face Sitting"     = performer sitting on another person's face — facesitting/queening position clearly shown
"Ass To Mouth"     = penis/toy goes directly from anus to mouth without cleaning — must be visually implied or shown
"Gagging"          = performer visibly gagging/choking on penis during oral sex — throat reaction, not a gag device. Different from "Gags" (which is a restraint device).
"Pussy Licking"    = requires TWO people — one performer performing oral sex on another performer's vulva/vagina. Evidence can be spread across frames (multi-angle or split-screen): e.g. one frame shows face buried between legs + another frame shows the vagina being licked — together these confirm the act. DO tag if the act is clearly happening even if not captured in a single frame. NEVER tag for: solo scenes, self-touching, fingering, face near thighs/stomach, kissing legs/body, blowjob scenes. When contact is genuinely unclear across ALL frames — do NOT tag.
"Rimjob"           = mouth/tongue clearly making contact with the anus. Must be unambiguous oral-to-anal contact — not anal penetration, not anal toys, not just buttocks licking.
"Deep Throat"      = penis inserted deeply into throat causing visible gagging/choking — full shaft disappears into mouth. Always also tag "Blowjob". Do NOT tag for regular blowjob.
"Titty Fucking"    = penis clearly inserted between breasts and thrusting — penis must be visible. Do NOT tag if breasts are merely being touched or groped.
"Handjob"          = hand stroking penis with a clear up-down motion — must show stroking, not just holding or touching.
"Footjob"          = foot/feet actively stroking or rubbing a penis — must show foot-to-penis contact with motion. Different from "Foot Fetish" (which is worship/licking).
"Pegging"          = woman penetrating a man anally with a strap-on
"Squirt"           = visible ejaculatory fluid jet/gush from vagina — must be a forceful visible spray or stream, not just wetness or lube. Do NOT tag for urination (use "Pissing") and do NOT tag for general vaginal wetness.
"Creampie"         = cum visibly dripping or oozing from vagina or anus after internal ejaculation — fluid must be clearly visible exiting the orifice. Always also tag "Cumshot".
"Facial"           = cum clearly visible on face (cheek, mouth, chin, forehead). Always also tag "Cumshot".
"Cum In Mouth"     = cum visibly deposited into open mouth — partner's mouth receives ejaculate. Always also tag "Cumshot". Different from "Swallow Cum" (which requires visible swallowing).
"Swallow Cum"      = performer visibly swallows cum — gulping/swallowing motion clearly shown after cum in mouth. Always also tag "Cumshot" and "Cum In Mouth".
"Cumshot"          = male ejaculation clearly visible anywhere (body, face, mouth, etc.). Always tag this when "Facial", "Creampie", "Cum In Mouth", or "Swallow Cum" is tagged. Do NOT tag based on performer reactions alone — ejaculation must be visually shown.
"Bukkake"          = multiple men ejaculating on one performer's face/body — requires 3+ men visibly cumming on same person
"Female Orgasm"    = female performer visibly orgasming — strong involuntary body reactions, clear climax moment
"Gaping"           = anus or vagina visibly stretched/gaping open after penetration — hole is clearly open and visible
"Double Anal Penetration" = TWO objects (penis/toy) penetrating the anus SIMULTANEOUSLY — both visibly inserted at same time
"Double Pussy Penetration" = TWO objects (penis/toy) penetrating the vagina SIMULTANEOUSLY — both visibly inserted at same time
"Glory Hole"       = performer receiving/giving sex through a hole in a wall — classic glory hole setup visible
"69 Pose"          = two performers performing mutual oral sex simultaneously in 69 position
"Striptease"       = performer slowly and deliberately removing clothing as a performance — strip dance or tease
"Spanking"         = clearly visible repeated slapping of buttocks
"Pissing"          = urination clearly shown as a sexual act (golden shower)
"Tickling"         = tickling used as a sexual/BDSM act
"Massage"          = massage scenario leading to sex, explicitly staged as massage
"Pumping toy"      = vacuum pump toy on genitals clearly shown in use
"Strap On"         = strap-on dildo used between female performers (not pegging)
"Toys"             = a sex toy (vibrator, dildo, butt plug, etc.) is visibly inserted or actively operated on a performer's genitals — must be clearly shown in use, not just held or lying nearby. A human penis is NOT a toy — NEVER tag if the only penetrating object visible is a penis.
"Fuck Machine"     = mechanical sex machine (automated thrusting device) clearly in use on performer
"Hardcore"         = explicitly rough/intense sex — vigorous pounding, aggressive action. Tag for clearly intense scenes.
"Exhibitionism"    = performer deliberately exposing themselves or having sex where others can see — intentional exposure, different from "Public" (which is about location risk)

═══ DOMINANCE / BDSM ═══
"Femdom"           = female actively dominating, male clearly submissive (not just confidence)
"BDSM"             = bondage, discipline, or sadomasochism elements clearly present
"Bondage"          = restraints on performer (ropes, chains, cuffs — general restraint)
"Shibari Bondage"  = Japanese rope bondage (decorative rope patterns clearly visible)
"Device Bondage"   = mechanical bondage device (stocks, spreader bars, racks, frames)
"Slave"            = slave/submissive roleplay — performer presented, treated, or referred to as a slave or property. Distinguished from "Slave Cage" (cage not required). Often combined with BDSM or Femdom.
"Slave Cage"       = performer confined inside a cage
"Handcuffs"        = metal handcuffs clearly visible on performer
"Clamps"           = nipple or body clamps clearly visible and attached
"Gags"             = mouth gag (ball gag, bit gag, tape) clearly in use
"Hanging Up"       = performer suspended/hung up by ropes from ceiling/structure
"Whipping"         = whip, flogger, or crop used to strike performer
"Electricity Play" = electric stimulation device (e-stim, violet wand) actively used
"Sensor Deprivation" = blindfold combined with additional sensory isolation (hood, earplugs)
"Wax"              = hot wax dripped onto body clearly shown
"Soap Play"        = soap/lather used as a fetish element (not just shower washing)
"Snuffing"         = staged death/execution roleplay (clearly fictional fantasy scenario)
"Humiliation"      = deliberate verbal or physical humiliation of performer — degrading acts, insults, spitting, used as an erotic element
"Ballbusting"      = deliberate strikes, kicks, or squeezing of male genitals as a BDSM/femdom act
"CBT"              = cock and ball torture — painful stimulation of male genitals (crushing, stretching, binding, needles)
"Enema"            = enema administered as a fetish/BDSM act — clearly shown
"Medical"          = medical fetish — clinical/hospital setting, medical instruments, doctor/nurse roleplay with explicit content
"CFNM"             = clothed female, naked male — woman(women) fully clothed while man is naked in sexual context

═══ SETTINGS & LOCATIONS ═══
"Outdoor"          = clearly outdoor setting (grass, forest, street, parking lot, etc.). Always tag "Outdoor" when "Beach" or "Public" in outdoor location is tagged.
"Public"           = clearly public place where strangers could see — can be outdoor (park, street) OR indoor (store, restaurant, stairwell). Bystanders visible OR real risk of discovery clearly implied. Tag "Outdoor" additionally if applicable.
"Beach"            = clearly at a beach — sand AND sea/ocean/waves must both be visible. Always also tag "Outdoor".
"Hotel"            = clearly hotel room setting — hotel bed, hotel decor, keycard door visible
"Office"           = clearly office or workplace setting (desk, computer, cubicle)
"Gym"              = clearly gym/fitness setting with visible exercise equipment
"Shower"           = sex taking place in a shower (running water, shower head visible)
"College"          = college/university setting — dorm room, campus, fraternity/sorority house, student context
"Party"            = party/club setting — multiple people, social gathering context, drinks, dancing
"Bathrooms"        = scene set in a bathroom with clearly visible bathroom fixtures (bathtub, sink, mirror, tiles). Use ONLY when the bathroom environment is the dominant setting. Do NOT tag if the scene is primarily a shower (use "Shower" instead). Do NOT tag for a brief glimpse of a bathroom.
"Toilet"           = sexual act on or near a toilet
"Sport"            = sporting or athletic activity context (field, court, locker room)
"Yoga"             = yoga poses or yoga mat used in a sexual context

═══ CAMERA STYLE ═══
"POV"              = point-of-view camera for most of video — viewer's perspective
"Vertical Video"   = portrait/vertical orientation (9:1у6 ratio, phone-style)
"Virtual Reality"  = VR format: fisheye lens, 180° or 360° dual-image side-by-side
"HD"               = clearly high-definition quality (sharp, 1080p or above)
"Vintage"          = clearly old recording (VHS grain, film artifacts, retro look)
"Voyeur"           = voyeuristic scenario where subject does not know they are being filmed — hidden camera angle, spy/peephole setup, or clearly staged hidden-cam roleplay. Do NOT tag for POV where the performer looks at camera — POV is consensual, Voyeur is covert/hidden.
"Close Up"         = extreme close-up camera shots — macro/detail shots of genitals or acts with no full body visible, tightly zoomed in on the action. Tag when close-up is the dominant shooting style.

═══ COSTUMES & ROLEPLAY ═══
"Cosplay"          = clearly identifiable pop-culture costume (anime, game, movie character)
"School Uniform"   = school uniform clothing clearly worn as fetish/roleplay
"Maid Uniform"     = French maid or cleaning maid costume clearly worn
"Nurse Uniform"    = nurse or medical costume clearly worn
"Cheerleader"      = cheerleader costume clearly worn — pom-poms, short skirt, sports uniform
"Secretary"        = secretary/office assistant roleplay — pencil skirt, blouse, glasses, boss/employee scenario
"Teacher"          = teacher/student roleplay — clearly staged classroom/authority scenario
"Babysitter"       = babysitter roleplay — clearly staged babysitter/employer scenario
"Military"         = military uniform or clearly military setting used in sexual context
"Uniform"          = any uniform not covered by specific categories (police, firefighter, flight attendant, etc.)
"Boots"            = boots prominently featured as a fetish focus (thigh-high, leather, etc.)
"High Heels"       = high heels prominently featured — worn during sex act or as fetish focus
"Lingerie"         = lingerie kept on and prominently featured during the sex act
"Stockings"        = stockings or pantyhose prominently featured and kept on
"Fishnet"          = fishnet stockings or fishnet clothing prominently featured
"Latex"            = latex or PVC clothing clearly worn by performer(s)
"Leather"          = leather clothing (corset, pants, jacket, harness) prominently worn as fetish
"Bikini"           = bikini worn during or just before sex — beach/pool bikini as sexual context
"Swimsuit"         = swimsuit/one-piece worn during or as part of sexual scenario
"Spandex"          = spandex/lycra tight clothing prominently featured
"Leggings"         = leggings prominently worn during sexual activity
"Tight Clothes"    = extremely tight/form-fitting clothing prominently featured as fetish focus
"Mask"             = mask worn by performer(s) — anonymous/mystery element, fetish mask
"Fantasy Character"= fantasy/mythological creature (elf, demon, vampire, fairy, centaur)
"Game Character"   = specific video game character costume (not generic cosplay)
"Step Fantasy"     = step-family roleplay scenario — stepmother/stepdaughter/stepbrother/stepsister/stepfather. IMPORTANT: must be clearly fictional roleplay, no real family implied.
"Behind The Scenes"= BTS content — camera crew visible, filming setup shown, making-of style
"Compilation"      = compilation video — multiple unrelated clips edited together, variety of scenes/performers
"Nudist"           = nudist/naturist context — nudity in non-sexual naturist setting (beach, resort)
"Wrestling"        = wrestling or physical combat used as sexual foreplay or mixed with sex

═══ ANIMATION & SPECIAL FORMATS ═══
"Anime"            = Japanese anime style animation (2D drawn, anime aesthetics)
"Hentai"           = explicit Japanese anime or manga style sexual content
"3d Animation"     = computer-generated 3D CGI animation (not anime/2D)
"Cartoon"          = Western cartoon or non-anime animation style
"Gameplay Video"   = actual video game footage — no real people, only in-game graphics
"Furry"            = anthropomorphic animal characters in sexual content
"Futanari"         = animated female character with a penis (anime/hentai context)
"Tentacle"         = tentacle content — animated or costume/prop

═══ OTHER FETISHES ═══
"Smoking"          = cigarette/cigar smoking used as a fetish element during sex
"Fetish"           = unusual fetish clearly present but not covered by any other category
"Transport Fetish" = sex taking place inside a moving/parked vehicle (car, train, plane)
"Insect Fetish"    = insects or bugs used as a fetish element
"Diaper"           = diaper fetish clearly shown (adult diaper worn/used as fetish)
"Funny"            = comedy or humorous content — bloopers, clearly awkward/comedic moments, intentional humor as the main tone. Do NOT tag for generic light-hearted content.
"Doggystyle"       = rear-entry penetration in doggy-style position — receiver on all fours or bent over, penetrator behind. Tag when this position is the clear dominant camera focus across multiple frames.
"Cowgirl"          = woman-on-top riding position — female astride male facing forward (cowgirl) or backward (reverse cowgirl). Tag when this position is the clear dominant camera focus across multiple frames.

═══ FORBIDDEN COMBINATIONS ═══
  NEVER "Titty Fucking" without a clearly visible penis
  NEVER "Double Penetration" without two penetrations visible SIMULTANEOUSLY
  NEVER "Gangbang" without also including "Group"
  NEVER "Solo" with "Group", "Threesome", "Gangbang", "Lesbian", or "Double Penetration"
  NEVER "Lesbian" if any male is actively participating in the scene
  NEVER "Bisexual Male" unless one male clearly has sex with both genders in same scene
  NEVER "Anal" if vaginal penetration is clearly shown and "Double Penetration" is not present — a penis is in vagina OR anus, not both at once
  NEVER "Hairy" and "Shaved" together for the same performer — they are mutually exclusive
  NEVER "Amateur" and "Pornstar" together — they are mutually exclusive
  WHEN IN DOUBT → OMIT

═══ COMPLETE CATEGORY LIST — copy names EXACTLY ═══
{cats_flat}

RULES:
• Copy names EXACTLY as written above. Case matters: "Femdom" not "femdom", "Big Tits" not "big tits"
• NEVER invent names not in the list. Wrong → correct: "Female Dominance"→"Femdom", "Dominatrix"→"Femdom", "Power Exchange"→"BDSM", "Multiple Partners"→"Group", "Oral Sex"→"Blowjob"
• Return ONLY a JSON array of 5–15 strings. No markdown, no explanation:
["Category1", "Category2", "Category3"]"""


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

3. PRIMARY TAGS — up to {tag_count} long-tail keyword phrases (3–6 words each). Include as many as are accurate and relevant — do not pad with low-quality tags to reach the limit.
   - Exact search queries real users type. Based on acts, appearance, setting.
   - Format: lowercase phrases, e.g. "amateur outdoor sex video"

4. SECONDARY TAGS — up to {secondary_tag_count} shorter keyword phrases (2–4 words). Include as many as are accurate and relevant — do not pad with low-quality tags to reach the limit.
   - Broader supporting search terms. Mix acts, appearance, category keywords.
   - Format: lowercase phrases

5. SEO DESCRIPTION — 2–3 short paragraphs (80–120 words total).
   - Natural readable prose, not keyword stuffing. Start with main action and performers.
   - Mention key visual details, setting, mood. Avoid the most extreme profanity but explicit terms are acceptable.
   - Keyword-rich but reads naturally. MUST NOT be empty — always generate this field.

--- OUTPUT ---
Return ONLY valid JSON, no markdown, no extra text. Stop immediately after the closing brace. All text in {language}.
IMPORTANT: primary_tags array must contain AT MOST {tag_count} items. secondary_tags array must contain AT MOST {secondary_tag_count} items. Do NOT generate more tags than requested.
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


_MAX_IMG_SIDE = 640  # max pixels per side before encoding — keeps image tokens ≤ 512/image

def pil_to_base64(img: Image.Image) -> str:
    img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > _MAX_IMG_SIDE:
        scale = _MAX_IMG_SIDE / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=82, optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def call_vision_model(
    prompt: str,
    images: List[Image.Image],
    sampling: Optional[Dict] = None,
    enable_thinking: bool = False,
    guided_json: Optional[Dict] = None,
    pass_name: str = "?",
) -> str:
    payload = {
        "prompt": prompt,
        "base64_images": [pil_to_base64(img) for img in images],
        "sampling_params": sampling or {"temperature": 0.65, "top_p": 0.90, "max_tokens": 2048},
        "enable_thinking": enable_thinking,
    }
    if guided_json:
        payload["guided_json"] = guided_json
    try:
        r = requests.post(MODEL_SERVER_URL, json=payload, timeout=420)
        r.raise_for_status()
        full_resp = r.json()
        out = full_resp.get("output", "").strip()
        finish_reason = full_resp.get("finish_reason", full_resp.get("stop_reason", "unknown"))
        max_tok = payload['sampling_params'].get('max_tokens')
        logger.info(f"[{pass_name}] response: {len(out)} chars, finish_reason={finish_reason}, max_tokens={max_tok}")
        if finish_reason in ("length", "max_tokens"):
            logger.warning(f"[{pass_name}] hit token limit (max_tokens={max_tok}) — response likely truncated")
        return out
    except Exception:
        logger.exception(f"[{pass_name}] model call failed")
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

    last_open  = text.rfind('{')
    last_close = text.rfind('}')
    logger.error(
        f"JSON parse failed. len={len(text)}, "
        f"last_open={last_open}, last_close={last_close}, "
        f"tail={repr(text[-80:])}"
    )
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


def _camel_to_words(s: str) -> str:
    """'MiddleEastern' → 'middle eastern', 'HomeVideo' → 'home video'"""
    spaced = re.sub(r'([A-Z][a-z]+)', r' \1', s).strip()
    return spaced.lower()

def _normalize_cats(raw: list) -> List[str]:
    result, seen = [], set()
    for cat in raw:
        if not isinstance(cat, str):
            continue
        key = cat.lower().strip()
        canonical = None

        # 1. Точное совпадение
        if key in ALLOWED_CATEGORIES_LOWER:
            canonical = ALLOWED_CATEGORIES_LOWER[key]
        # 2. Alias-карта
        elif key in _CAT_ALIASES:
            canonical = _CAT_ALIASES[key]
            logger.debug(f"[normalize_cats] alias '{cat}' → '{canonical}'")
        else:
            # 3. CamelCase → слова → повтор поиска ("MiddleEastern" → "middle eastern")
            key2 = _camel_to_words(cat)
            if key2 in ALLOWED_CATEGORIES_LOWER:
                canonical = ALLOWED_CATEGORIES_LOWER[key2]
                logger.debug(f"[normalize_cats] camel '{cat}' → '{canonical}'")
            elif key2 in _CAT_ALIASES:
                canonical = _CAT_ALIASES[key2]
                logger.debug(f"[normalize_cats] camel+alias '{cat}' → '{canonical}'")
            else:
                # 4. N-грамм поиск по словам ("GaggedPerson" → "gagged person" → слова ["gagged","person"]
                #    ищем все подпоследовательности от длинных к коротким)
                words = key2.split()
                for n in range(len(words), 0, -1):
                    if canonical:
                        break
                    for i in range(len(words) - n + 1):
                        phrase = " ".join(words[i:i+n])
                        if phrase in ALLOWED_CATEGORIES_LOWER:
                            canonical = ALLOWED_CATEGORIES_LOWER[phrase]
                            logger.debug(f"[normalize_cats] ngram '{cat}' → '{canonical}'")
                            break
                        elif phrase in _CAT_ALIASES:
                            canonical = _CAT_ALIASES[phrase]
                            logger.debug(f"[normalize_cats] ngram+alias '{cat}' → '{canonical}'")
                            break

        if canonical is None:
            logger.debug(f"[normalize_cats] unknown: '{cat}'")
            continue
        if canonical.lower() not in seen:
            result.append(canonical)
            seen.add(canonical.lower())
    return result


def validate_categories(categories: List[str], orientation: Optional[str]) -> List[str]:
    cats = list(categories)

    def cl():
        return {c.lower() for c in cats}

    def remove(*names: str):
        low = {n.lower() for n in names}
        nonlocal cats
        cats = [c for c in cats if c.lower() not in low]

    requires_penis  = {"titty fucking", "handjob", "cumshot", "facial", "creampie", "big dick"}
    requires_vagina = {"pussy licking", "squirt", "creampie"}
    # Tags that only make sense for real human performers
    real_only = {"amateur", "pornstar", "webcam", "onlyfans", "casting", "crossdresser"}
    # Animation format tags
    animation_types = {"3d animation", "anime", "cartoon", "hentai", "furry", "futanari", "tentacle"}

    # ── 1. Orientation-based exclusions ───────────────────────────────────────
    if orientation == "gay":
        remove(*requires_vagina, "lesbian", "milf", "squirt", "pregnant", "lactating",
               "pussy licking", "pegging", "strap on")

    elif orientation == "shemale":
        remove("lesbian", "gay", "pegging", "strap on", "pregnant",
               "lactating", "pussy licking", "squirt")

    elif orientation == "straight":
        remove("gay")
        if "lesbian" in cl():
            remove(*requires_penis, "pegging")  # Lesbian = no male → no penis acts, no pegging

    # ── 2. Performer count logic ───────────────────────────────────────────────
    cats_l = cl()

    # Gangbang always implies Group — auto-add Group if missing
    if "gangbang" in cats_l and "group" not in cats_l:
        cats.append("Group")

    cats_l = cl()

    # Threesome (exactly 3) and Group (4+) are mutually exclusive — keep Group
    if "threesome" in cats_l and "group" in cats_l:
        remove("Threesome")

    cats_l = cl()

    # Bisexual Male requires male participants → incompatible with Lesbian (all-female)
    if "bisexual male" in cats_l and "lesbian" in cats_l:
        remove("Lesbian")

    cats_l = cl()

    # Bisexual Male requires a group context (3+ people)
    if "bisexual male" in cats_l and not any(x in cats_l for x in ("group", "threesome", "gangbang")):
        remove("Bisexual Male")

    cats_l = cl()

    # Solo excludes all multi-person and partner-dependent tags
    if "solo" in cats_l:
        remove("Group", "Threesome", "Gangbang", "Double Penetration", "Anal",
               "Bisexual Male", "Lesbian", "Cuckold", "Pussy Licking",
               "Rimjob", "Handjob", "Titty Fucking", "Pegging", "Strap On",
               "Fisting", "Cumshot", "Facial", "Creampie", "Casting",
               "Interracial", "Old/Young", "Bisexual Male")

    # ── 3. Role / dominance exclusions ───────────────────────────────────────
    cats_l = cl()

    # Femdom has no meaning in all-male or all-female scenes
    if orientation == "gay" or "lesbian" in cats_l:
        remove("Femdom")

    # ── 4. Format / production mutual exclusions ──────────────────────────────
    cats_l = cl()

    # Japanese Censored ↔ Uncensored — keep Censored (more conservative)
    if "japanese censored" in cats_l and "japanese uncensored" in cats_l:
        remove("Japanese Uncensored")

    cats_l = cl()

    # Amateur ↔ Pornstar — mutually exclusive; Pornstar wins (Amateur is over-tagged)
    if "amateur" in cats_l and "pornstar" in cats_l:
        remove("Amateur")

    cats_l = cl()

    # Vintage is incompatible with modern production styles and formats
    if "vintage" in cats_l:
        remove("HD", "Virtual Reality", "Vertical Video", "OnlyFans", "Webcam")

    cats_l = cl()

    # Virtual Reality and Vertical Video are different camera formats
    if "virtual reality" in cats_l and "vertical video" in cats_l:
        remove("Vertical Video")

    # ── 5. Anal / vaginal mutual exclusion ───────────────────────────────────
    cats_l = cl()

    # A penis occupies one orifice at a time.
    # If vaginal indicators (Creampie or Squirt) are present without Double Penetration → not anal
    vaginal_indicators = {"creampie", "squirt", "pussy licking"}
    if ("anal" in cats_l
            and "double penetration" not in cats_l
            and vaginal_indicators & cats_l):
        remove("Anal")

    # ── 6. Animation / rendered content exclusions ───────────────────────────
    cats_l = cl()

    if "gameplay video" in cats_l:
        # Pure in-game footage: only format/technical tags are valid
        keep = {"gameplay video", "hd", "vertical video", "virtual reality",
                "vintage", "anime", "cartoon", "3d animation"}
        cats = [c for c in cats if c.lower() in keep]

    elif any(a in cats_l for a in animation_types):
        # If strong real-performer evidence exists alongside animation tags → animation is a false
        # positive (e.g. model read a watermark like "hentaied.com"). Remove animation tags.
        real_performer_indicators = {"pornstar", "amateur", "blonde", "brunette", "red head",
                                     "big tits", "big ass", "big dick", "teen", "mature", "milf",
                                     "granny", "oiled", "tattoo", "hairy", "bbw", "chubby"}
        if real_performer_indicators & cats_l:
            remove("Anime", "Hentai", "Cartoon", "3d Animation", "Furry", "Futanari", "Tentacle")
        else:
            # Confirmed animation: remove real-performer production tags
            remove(*real_only)

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

        logger.info(f"Pass1 start: frames={len(frames_1a)} lang={language}")
        raw1 = call_vision_model(
            build_analysis_prompt(len(frames_1a), ts_map, desc_style, language),
            frames_1a,
            {"temperature": 0.45, "top_p": 0.85, "max_tokens": 2000},
            pass_name="Pass1",
        )
        p1 = extract_json_from_response(raw1)
        if not p1:
            return {"status": "error", "reason": "pass1 invalid response"}

        description = _redact_blocked(p1.get("description", "").strip())
        orientation = p1.get("orientation", "straight")
        if orientation not in VALID_ORIENTATIONS:
            orientation = "straight"
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

        # ── Pass 2: Categories ────────────────────────────────────────────────
        logger.info(f"Pass2-cats start: orient={orientation}")
        raw2 = call_vision_model(
            build_categories_prompt(len(frames_1a), orientation, description),
            frames_1a,
            {"temperature": 0.15, "top_p": 0.80, "max_tokens": 500, "repetition_penalty": 1.3},
            pass_name="Pass2-cats",
        )
        cats_raw_list: List[str] = []
        logger.info(f"[Pass2-cats] raw: {repr(raw2[:300]) if raw2 else 'EMPTY'}")
        if raw2:
            raw2_clean = re.sub(r'<think>[\s\S]*?</think>', '', raw2, flags=re.IGNORECASE).strip()
            if not raw2_clean:
                raw2_clean = raw2
            # Шаг 1: ищем JSON-массив строк ["Cat1", "Cat2", ...]
            try:
                m_arr = re.search(r'\[[\s\S]*?\]', raw2_clean)
                if not m_arr:
                    # Обрезанный массив — добавляем закрывающую скобку
                    m_open = re.search(r'\[', raw2_clean)
                    if m_open:
                        fragment = re.sub(r',\s*$', '', raw2_clean[m_open.start():].strip()) + ']'
                        m_arr_str = fragment
                    else:
                        m_arr_str = None
                else:
                    m_arr_str = m_arr.group()
                if m_arr_str:
                    parsed2 = json.loads(m_arr_str)
                    if isinstance(parsed2, list):
                        cats_raw_list = _normalize_cats([str(c) for c in parsed2 if isinstance(c, str)])
            except Exception:
                pass
            # Шаг 2: нормализация не дала результат — ищем имена в кавычках по всему тексту
            if not cats_raw_list:
                logger.warning("[Pass2-cats] нормализация пуста — ищем имена в тексте")
                candidates = re.findall(r"['\"]([A-Za-z][A-Za-z0-9 /\-]{1,40})['\"]", raw2_clean)
                cats_raw_list = _normalize_cats(candidates)
            # Шаг 3: последний шанс — слова с большой буквы
            if not cats_raw_list:
                cats_raw_list = _normalize_cats(re.findall(r'\b([A-Z][A-Za-z0-9 /]{2,30})\b', raw2_clean))
        # Дедупликация (сохранить порядок)
        seen_cats: set = set()
        cats_deduped: List[str] = []
        for c in cats_raw_list[:10]:
            if c.lower() not in seen_cats:
                cats_deduped.append(c)
                seen_cats.add(c.lower())
        cats_raw = _filter_blocked_list(cats_deduped)

        final_categories = validate_categories(cats_raw, orientation)
        logger.info(f"Pass 1+2 done: orient={orientation} cats={final_categories} watermarks={watermarks} scenes={len(key_scenes)}")

        # Screenshot selection disabled
        top5 = []
        thumb_frame = None

        # ── Pass SEO (multi-language) ────────────────────────────────────────
        final_categories = final_categories[:category_count]
        # SEO pass is text-only: description and categories are already in the prompt
        seo_ref: List = []

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
        _SEO_SCHEMA = {
            "type": "object",
            "properties": {
                "meta_title":       {"type": "string"},
                "meta_description": {"type": "string"},
                "seo_description":  {"type": "string"},
                "primary_tags":     {"type": "array", "items": {"type": "string"}},
                "secondary_tags":   {"type": "array", "items": {"type": "string"}},
            },
            "required": ["meta_title", "meta_description", "seo_description", "primary_tags", "secondary_tags"],
        }
        logger.info(f"SEO-{p_lang_code} start: cats={len(final_categories)}")
        raw_seo = call_vision_model(
            build_seo_prompt(description[:500], final_categories, orientation, p_lang_name,
                             tag_count, secondary_tag_count),
            seo_ref,
            {"temperature": 0.3, "top_p": 0.85, "max_tokens": 4096, "repetition_penalty": 1.3},
            guided_json=_SEO_SCHEMA,
            pass_name=f"SEO-{p_lang_code}",
        )
        p_seo = extract_json_from_response(raw_seo) or _seo_fallback(raw_seo or "")
        primary_tags   = _filter_blocked_list([t.strip() for t in p_seo.get("primary_tags", []) if t.strip()][:tag_count])
        secondary_tags = _filter_blocked_list([t.strip() for t in p_seo.get("secondary_tags", []) if t.strip()][:secondary_tag_count])
        seo_desc = _redact_blocked(p_seo.get("seo_description", "").strip())
        # Fallback: if seo_description empty, use meta_description
        if not seo_desc:
            seo_desc = _redact_blocked(p_seo.get("meta_description", "").strip())
            if seo_desc:
                logger.warning("seo_description was empty — using meta_description as fallback")
        seo_by_lang[p_lang_code] = {
            "meta_title":       _redact_blocked(p_seo.get("meta_title", "").strip()),
            "meta_description": _redact_blocked(p_seo.get("meta_description", "").strip()),
            "seo_description":  seo_desc,
        }
        logger.info(f"Pass SEO [{p_lang_code}]: title={len(seo_by_lang[p_lang_code]['meta_title'])}")

        # Extra languages — translate only meta_title, meta_description, seo_description
        base_title    = seo_by_lang[p_lang_code]["meta_title"]
        base_meta     = seo_by_lang[p_lang_code]["meta_description"]
        base_seo_desc = seo_by_lang[p_lang_code]["seo_description"]
        for lang_code, lang_name in all_langs[1:]:
            _SEO_TR_SCHEMA = {
                "type": "object",
                "properties": {
                    "meta_title":       {"type": "string"},
                    "meta_description": {"type": "string"},
                    "seo_description":  {"type": "string"},
                },
                "required": ["meta_title", "meta_description", "seo_description"],
            }
            logger.info(f"SEO-tr-{lang_code} start")
            raw_tr = call_vision_model(
                build_seo_translate_prompt(base_title, base_meta, base_seo_desc, lang_name),
                seo_ref,
                {"temperature": 0.2, "top_p": 0.80, "max_tokens": 2048},
                guided_json=_SEO_TR_SCHEMA,
                pass_name=f"SEO-tr-{lang_code}",
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

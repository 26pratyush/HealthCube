from datetime import datetime
import json, re, pathlib
from slugify import slugify
from rapidfuzz import process, fuzz
from pymongo import MongoClient, ASCENDING, DESCENDING, TEXT, UpdateOne

MONGO_URI = "mongodb://localhost:27017"
DB_NAME   = "symptosense"

def now():
    return datetime.utcnow()

def canon_id(text: str) -> str:

    t = text.strip().lower()
    t = re.sub(r"[^\w]+", "_", t)              # non-word → _
    t = re.sub(r"_+", "_", t).strip("_")
    return t

def split_or_terms(text: str):
    """
    Split patterns like 'nose or gum bleeding' or 'rash or rose spots'
    into individual normalized terms.
    """
    if " or " in text:
        parts = [p.strip() for p in re.split(r"\bor\b", text)]
        tokens = text.split()
        if len(tokens) >= 3:
            last = tokens[-1]
            return [p if p.endswith(last) else f"{p} {last}".strip() for p in parts]
        return parts
    return [text.strip()]

def to_canonical_list(items):
    out = []
    for s in items:
        for term in split_or_terms(s.replace("_", " ").replace("-", " ")):
            out.append(canon_id(term))
    return sorted(set(out))

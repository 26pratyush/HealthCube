from pymongo import MongoClient, UpdateOne
from collections import defaultdict
import json, pathlib
from step1_setup import MONGO_URI, DB_NAME, now, canon_id, to_canonical_list

client = MongoClient(MONGO_URI)
db = client[DB_NAME]

# Paths (adjust if needed)
JSON_PATH = pathlib.Path("disease_symptoms_categorized.json")
TXT_PATH  = pathlib.Path("Real_symptoms.txt")

# 1) Collect symptom IDs from the JSON buckets
data = json.loads(JSON_PATH.read_text(encoding="utf-8"))
bucket_symptoms = set()
for disease, buckets in data.items():
    for bucket, arr in buckets.items():
        if isinstance(arr, list):
            bucket_symptoms.update(to_canonical_list(arr))

# 2) Build synonyms from the TXT free-form list (per disease lines)
raw_terms = set()
for line in TXT_PATH.read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if not line or ":" not in line:
        continue
    _, rhs = line.split(":", 1)
    # split commas, normalize
    for piece in rhs.split(","):
        term = piece.strip()
        if term:
            raw_terms.add(term)

# map raw terms to canonical IDs
canon_to_syns = defaultdict(set)
for term in raw_terms:
    for c in to_canonical_list([term]):
        canon_to_syns[c].add(term.strip().lower())

for c in bucket_symptoms:
    canon_to_syns[c].add(c.replace("_", " "))

# 3) Upsert into `symptoms`
ops = []
for sid, syns in canon_to_syns.items():
    doc = {
        "_id": sid,
        "label": " ".join([w.capitalize() if i==0 else w for i,w in enumerate(sid.replace("_"," ").split())]),
        "synonyms": sorted(set(syns)),
        "value_type": "bool",
        "units": None,
        "notes": "",
        "updated_at": now()
    }
    ops.append(UpdateOne({"_id": sid},
                         {"$setOnInsert": {"created_at": now(), "version": 1},
                          "$set": doc},
                         upsert=True))

if ops:
    res = db.symptoms.bulk_write(ops, ordered=False)
    print("Symptoms upserted:", res.upserted_count + res.modified_count)

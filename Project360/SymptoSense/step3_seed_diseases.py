from pymongo import MongoClient, UpdateOne
from step1_setup import MONGO_URI, DB_NAME, now, to_canonical_list
import json, pathlib

client = MongoClient(MONGO_URI)
db = client[DB_NAME]

JSON_PATH = pathlib.Path("disease_symptoms_categorized.json")
data = json.loads(JSON_PATH.read_text(encoding="utf-8"))

# Default bucket weights (tune later)
W = {"hallmark": 3.0, "core": 2.0, "associated": 1.0, "general": 0.5}

ops = []
for name, buckets in data.items():
    disease_id = to_canonical_list([name])[0]
    profile = {"hallmark": [], "core": [], "associated": [], "general": [], "exclude_if":[]}

    for bucket, arr in buckets.items():
        if bucket not in profile:      # ignore unknown keys
            continue
        for sid in to_canonical_list(arr):
            profile[bucket].append({"symptom_id": sid, "weight": W.get(bucket, 1.0)})

    doc = {
        "_id": disease_id,
        "name": name,
        "aliases": [],
        "category_tags": [],
        "symptom_profile": profile,
        "runtime_stats": {"cases": 0, "symptom_counts": {}},
        "priors": {"region_bias": [], "prevalence_score": 0.5},
        "updated_at": now()
    }

    ops.append(UpdateOne({"_id": disease_id},
                         {"$setOnInsert": {"created_at": now(), "version": 1},
                          "$set": doc},
                         upsert=True))

if ops:
    res = db.diseases.bulk_write(ops, ordered=False)
    print("Diseases upserted:", res.upserted_count + res.modified_count)

# step9_catalog_ops.py
from pymongo import MongoClient, UpdateOne
from datetime import datetime
import re

MONGO_URI = "mongodb://localhost:27017"
DB_NAME   = "symptosense"

client = MongoClient(MONGO_URI)
db = client[DB_NAME]

def utc_now(): return datetime.utcnow()
def canon_id(text:str)->str:
    t = text.strip().lower()
    t = re.sub(r"[^\w]+","_", t)
    t = re.sub(r"_+","_", t).strip("_")
    return t

# ---------- SYMPTOMS ----------
def add_symptom(label:str, synonyms:list[str]|None=None,
                value_type:str="bool", units:str|None=None, notes:str="") -> str:

    sid = canon_id(label)
    doc = {
        "_id": sid,
        "label": label.strip(),
        "synonyms": sorted(set([s.lower().strip() for s in (synonyms or [])] + [label.lower().strip()])),
        "value_type": value_type,
        "units": units,
        "notes": notes,
        "updated_at": utc_now()
    }
    db.symptoms.update_one({"_id": sid},
                           {"$setOnInsert": {"created_at": utc_now(), "version": 1},
                            "$set": doc},
                           upsert=True)
    return sid

def add_synonyms(symptom_id:str, more_synonyms:list[str]):
    db.symptoms.update_one({"_id": symptom_id},
                           {"$addToSet": {"synonyms": {"$each": [s.lower().strip() for s in more_synonyms]}}})

# ---------- DISEASES ----------
DEFAULT_WEIGHTS = {"hallmark": 3.0, "core": 2.0, "associated": 1.0, "general": 0.5}

def upsert_disease(name:str,
                   hallmark:list[str]|None=None,
                   core:list[str]|None=None,
                   associated:list[str]|None=None,
                   general:list[str]|None=None,
                   aliases:list[str]|None=None,
                   category_tags:list[str]|None=None,
                   prevalence_score:float=0.5) -> str:
    did = canon_id(name)

    def _bucket(sym_list, bucket):
        out = []
        for term in (sym_list or []):
            sid = term if db.symptoms.find_one({"_id": term}) else add_symptom(term)
            out.append({"symptom_id": sid, "weight": DEFAULT_WEIGHTS[bucket]})
        return out

    doc = {
        "_id": did,
        "name": name.strip(),
        "aliases": aliases or [],
        "category_tags": category_tags or [],
        "symptom_profile": {
            "hallmark":  _bucket(hallmark,  "hallmark"),
            "core":      _bucket(core,      "core"),
            "associated":_bucket(associated,"associated"),
            "general":   _bucket(general,   "general"),
            "exclude_if": []
        },
        "runtime_stats": {"cases": 0, "symptom_counts": {}},
        "priors": {"region_bias": [], "prevalence_score": float(prevalence_score)},
        "updated_at": utc_now()
    }

    db.diseases.update_one({"_id": did},
                           {"$setOnInsert": {"created_at": utc_now(), "version": 1},
                            "$set": doc},
                           upsert=True)
    return did

def attach_symptom_to_disease(disease_id:str, bucket:str, symptom_term:str, weight:float|None=None):
    if bucket not in {"hallmark","core","associated","general","exclude_if"}:
        raise ValueError("invalid bucket")

    # ensure disease exists
    if not db.diseases.find_one({"_id": disease_id}, {"_id":1}):
        raise ValueError("disease not found")

    if bucket == "exclude_if":
        sid = symptom_term if db.symptoms.find_one({"_id": symptom_term}) else add_symptom(symptom_term)
        db.diseases.update_one({"_id": disease_id},
                               {"$addToSet": {f"symptom_profile.exclude_if": sid},
                                "$set": {"updated_at": utc_now()}})
        return sid

    # for weighted buckets
    sid = symptom_term if db.symptoms.find_one({"_id": symptom_term}) else add_symptom(symptom_term)
    w   = float(weight if weight is not None else DEFAULT_WEIGHTS[bucket])

    # prevent duplicates (by symptom_id) inside the bucket
    d = db.diseases.find_one({"_id": disease_id}, {f"symptom_profile.{bucket}":1})
    existing = {x["symptom_id"] for x in (d or {}).get("symptom_profile", {}).get(bucket, [])}
    if sid in existing:
        return sid

    db.diseases.update_one({"_id": disease_id},
                           {"$push": {f"symptom_profile.{bucket}": {"symptom_id": sid, "weight": w}},
                            "$set": {"updated_at": utc_now()}})
    return sid

if __name__ == "__main__": #demo:
    # 1) add a fresh symptom with synonyms
    s_id = add_symptom("Pain behind the eyes", ["retro-orbital pain", "retro orbital pain"])
    print("symptom:", s_id)

    # 2) create/update a disease quickly
    d_id = upsert_disease(
        "Dengue",
        hallmark=["Rash", "Nose bleeding", "Gum bleeding", "Pain behind the eyes"],
        core=["High fever", "Severe headache", "Joint pain", "Muscle pain"],
        associated=["Nausea", "Vomiting", "Fatigue"],
        general=[]
    )
    print("disease:", d_id)

    # 3) attach a new symptom to an existing disease
    sid2 = attach_symptom_to_disease("dengue", "associated", "Loss of appetite", weight=1.0)
    print("attached:", sid2)

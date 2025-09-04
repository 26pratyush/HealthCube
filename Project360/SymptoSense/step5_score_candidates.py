from pymongo import MongoClient
from collections import defaultdict
import argparse
from datetime import datetime

MONGO_URI = "mongodb://localhost:27017"
DB_NAME   = "symptosense"

# ----------------- helpers -----------------
import re
def canon_id(text: str) -> str:
    t = text.strip().lower()
    t = re.sub(r"[^\w]+", "_", t)
    t = re.sub(r"_+", "_", t).strip("_")
    return t

# Connect
client = MongoClient(MONGO_URI)
db = client[DB_NAME]

def load_disease_symptom_id_set():
    """Collect all symptom_ids that appear in any disease bucket."""
    ids = set()
    for d in db.diseases.find({}, {"symptom_profile": 1}):
        sp = d.get("symptom_profile", {})
        for bucket in ("hallmark","core","associated","general"):
            for itm in sp.get(bucket, []):
                sid = itm.get("symptom_id")
                if sid: ids.add(sid)
    return ids

DISEASE_SYMPTOM_IDS = load_disease_symptom_id_set()

def resolve_symptom_terms(terms):
    out = set()
    for t in terms:
        t_norm = t.strip().lower()

        # Try as canonical _id
        row = db.symptoms.find_one({"_id": t_norm}, {"_id": 1})
        if row:
            out.add(row["_id"])
            continue

        # Try exact synonym
        row = db.symptoms.find_one({"synonyms": t_norm}, {"_id": 1})
        if row:
            out.add(row["_id"])
            continue

        # Fallback: text search
        try:
            row = db.symptoms.find_one(
                {"$text": {"$search": t_norm}},
                {"_id": 1, "score": {"$meta": "textScore"}}
            )
            if row:
                out.add(row["_id"])
                continue
        except Exception:
            pass

        # Last resort: canonicalize
        out.add(canon_id(t_norm))

    # Map any non-used IDs to a used one when possible
    mapped = set()
    for sid in out:
        if sid in DISEASE_SYMPTOM_IDS:
            mapped.add(sid)
            continue
        # Try to find a symptom doc for human phrase -> id that is used
        phrase = sid.replace("_", " ")
        # try exact synonym match to a used id
        row = db.symptoms.find_one(
            {"synonyms": phrase},
            {"_id": 1}
        )
        if row and row["_id"] in DISEASE_SYMPTOM_IDS:
            mapped.add(row["_id"])
            continue

        # As a final attempt, try text search to find a used id
        try:
            hits = list(db.symptoms.find(
                {"$text": {"$search": phrase}},
                {"_id": 1, "score": {"$meta": "textScore"}}
            ).sort([("score", -1)]).limit(5))
            found = None
            for h in hits:
                if h["_id"] in DISEASE_SYMPTOM_IDS:
                    found = h["_id"]; break
            mapped.add(found if found else sid)
        except Exception:
            mapped.add(sid)

    return mapped

# ----------------- scoring -----------------
def score_candidates(selected_present: set[str], selected_absent: set[str] = None, top_k=5):
    selected_absent = selected_absent or set()
    pipeline = [
        {"$project": {
            "name": 1,
            "symptom_profile": 1,
            "all": {"$concatArrays": [
                {"$ifNull": ["$symptom_profile.hallmark",   []]},
                {"$ifNull": ["$symptom_profile.core",       []]},
                {"$ifNull": ["$symptom_profile.associated", []]},
                {"$ifNull": ["$symptom_profile.general",    []]}
            ]}
        }},
        {"$addFields": {
            "present_sum": {
                "$sum": {"$map": {"input": "$all", "as": "s",
                    "in": {"$cond":[{"$in":["$$s.symptom_id", list(selected_present)]}, "$$s.weight", 0]}}}
            },
            "absent_penalty": {
                "$sum": {"$map": {"input": "$all", "as": "s",
                    "in": {"$cond":[{"$in":["$$s.symptom_id", list(selected_absent)]}, {"$multiply":["$$s.weight", 0.5]}, 0]}}}
            },
            # count hallmark/core hits separately
            "core_hits": {
                "$size": {
                    "$setIntersection": [
                        {"$map": {"input": {"$concatArrays":[
                            {"$ifNull":["$symptom_profile.hallmark", []]},
                            {"$ifNull":["$symptom_profile.core", []]}
                        ]}, "as": "c", "in": "$$c.symptom_id"}},
                        list(selected_present)
                    ]
                }
            },
            "core_weight": {
                "$sum": {"$map": {"input": {"$concatArrays":[
                    {"$ifNull":["$symptom_profile.hallmark", []]},
                    {"$ifNull":["$symptom_profile.core", []]}
                ]}, "as": "c",
                    "in": {"$cond":[{"$in":["$$c.symptom_id", list(selected_present)]}, "$$c.weight", 0]}}}
            }
        }},
        {"$addFields": {"score": {"$subtract": ["$present_sum", "$absent_penalty"]}}},
        # Optional: discard diseases with no core/hallmark evidence (tunable)
        {"$match": {"core_hits": {"$gte": 1}}},  # <- change to 1 if you want stricter gating (already changed)
        {"$sort": {"score": -1, "present_sum": -1}},
        {"$limit": top_k}
    ]
    return list(db.diseases.aggregate(pipeline))

def explain_candidates(cands, present_ids, absent_ids):
    explained = []
    for d in cands:
        contrib_pos, contrib_neg = [], []
        sp = d.get("symptom_profile", {})
        for bucket in ("hallmark", "core", "associated", "general"):
            for itm in sp.get(bucket, []):
                sid, w = itm.get("symptom_id"), itm.get("weight", 0)
                if not sid: continue
                if sid in present_ids:
                    contrib_pos.append((sid, w, bucket))
                elif sid in absent_ids:
                    contrib_neg.append((sid, -0.5 * w, bucket))
        contrib_pos.sort(key=lambda x: -x[1])
        contrib_neg.sort(key=lambda x: x[1])
        explained.append({
            "_id": d.get("_id"),
            "name": d.get("name", ""),
            "score": round(d.get("score", 0.0), 3),
            "present_sum": round(d.get("present_sum", 0.0), 3),
            "absent_penalty": round(d.get("absent_penalty", 0.0), 3),
            "top_support": contrib_pos[:5],
            "top_penalty": contrib_neg[:3],
        })
    return explained

# ----------------- CLI -----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score disease candidates from selected symptoms.")
    parser.add_argument("--sym", type=str, default="", help="Comma-separated symptom terms present (canonical IDs or synonyms).")
    parser.add_argument("--abs", type=str, default="", help="Comma-separated symptom terms explicitly absent.")
    parser.add_argument("--k", type=int, default=5, help="Top-K diseases to return.")
    args = parser.parse_args()

    print(f"DB: {DB_NAME}")
    print("Counts:",
          "symptoms =", db.symptoms.estimated_document_count(),
          "| diseases =", db.diseases.estimated_document_count())

    present_terms = [s.strip() for s in args.sym.split(",") if s.strip()]
    absent_terms  = [s.strip() for s in args.abs.split(",") if s.strip()]

    present_ids = resolve_symptom_terms(present_terms)
    absent_ids  = resolve_symptom_terms(absent_terms)

    print("\nResolved symptom IDs:")
    print(" present:", sorted(present_ids))
    print(" absent :", sorted(absent_ids))

    cands = score_candidates(set(present_ids), set(absent_ids), top_k=args.k)

    if not cands:
        print("\nNo candidates returned. Sanity check: Did Steps 2 & 3 insert data?")
    else:
        explained = explain_candidates(cands, present_ids, absent_ids)
        print("\nTop candidates:")
        for i, d in enumerate(explained, 1):
            print(f"\n{i}) {d.get('name','(unnamed)')}  | score={d['score']}  "
                  f"(+{d['present_sum']}, {d['absent_penalty']})")
            if d["top_support"]:
                print("   supporting:", ", ".join([f"{sid}({w})" for sid,w,_ in d["top_support"]]))
            if d["top_penalty"]:
                print("   penalty   :", ", ".join([f"{sid}({w})" for sid,w,_ in d["top_penalty"]]))

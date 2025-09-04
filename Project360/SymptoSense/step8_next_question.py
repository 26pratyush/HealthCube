from pymongo import MongoClient
import argparse, re

MONGO_URI = "mongodb://localhost:27017"
DB_NAME   = "symptosense"

client = MongoClient(MONGO_URI)
db = client[DB_NAME]

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
            }
        }},
        {"$addFields": {"score": {"$subtract": ["$present_sum", "$absent_penalty"]}}},
        {"$sort": {"score": -1, "present_sum": -1}},
        {"$limit": top_k}
    ]
    return list(db.diseases.aggregate(pipeline))

def next_question(present_ids:set[str], absent_ids:set[str], asked_ids:set[str], top_k:int=5):
    # 1) shortlist diseases
    cands = score_candidates(present_ids, absent_ids, top_k=top_k)
    if not cands:
        return None, [], []

    # 2) compile candidate symptom weights across top diseases
    weight_map = {}
    for d in cands:
        sp = d.get("symptom_profile", {})
        local = {}
        for bucket in ("hallmark","core","associated","general"):
            for itm in sp.get(bucket, []):
                sid = itm.get("symptom_id"); w = float(itm.get("weight", 0))
                if sid: local[sid] = w
        for sid in set(local.keys()) | set(weight_map.keys()):
            weight_map.setdefault(sid, []).append(local.get(sid, 0.0))

    # 3) score each not-asked symptom by variance across top diseases
    from statistics import pstdev
    candidates = []
    for sid, ws in weight_map.items():
        if sid in present_ids or sid in absent_ids or sid in asked_ids:
            continue
        # skip totally non-informative (all zeros)
        if all(w == 0 for w in ws): 
            continue
        var = pstdev(ws) if len(ws) > 1 else 0.0
        qdoc = db.questions.find_one({"symptom_id": sid}, {"is_red_flag":1, "prompt":1})
        prompt = (qdoc or {}).get("prompt", f"Do you have {sid.replace('_',' ')}?")
        bump = 0.25 if (qdoc and qdoc.get("is_red_flag")) else 0.0
        score = var + bump
        candidates.append((sid, prompt, score))

    if not candidates:
        return None, cands, []

    candidates.sort(key=lambda x: x[2], reverse=True)
    best = candidates[0]  # (sid, prompt, score)
    return best, cands, candidates[:10]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--present", type=str, default="", help="Comma-separated symptom_ids present")
    ap.add_argument("--absent",  type=str, default="", help="Comma-separated symptom_ids absent")
    ap.add_argument("--asked",   type=str, default="", help="Comma-separated symptom_ids already asked")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    present = set([s.strip() for s in args.present.split(",") if s.strip()])
    absent  = set([s.strip() for s in args.absent.split(",") if s.strip()])
    asked   = set([s.strip() for s in args.asked.split(",") if s.strip()])

    best, cands, top_symptoms = next_question(present, absent, asked, top_k=args.k)

    print("Top diseases (after current answers):")
    for i, d in enumerate(cands, 1):
        print(f"{i:2d}. {d.get('name','')}  score={d.get('score',0):.2f}")

    if best:
        sid, prompt, score = best
        print("\nNext question to ask:")
        print(f"- {prompt}  [{sid}]  (info≈{score:.3f})")
        print("\nOther strong candidates:")
        for sid, prompt, sc in top_symptoms[1:]:
            print(f"- {prompt}  [{sid}]  (info≈{sc:.3f})")
    else:
        print("\nNo further informative question found (either confident or no unseen symptoms).")

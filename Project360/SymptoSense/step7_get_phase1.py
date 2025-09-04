from pymongo import MongoClient, DESCENDING
import argparse

MONGO_URI = "mongodb://localhost:27017"
DB_NAME   = "symptosense"

client = MongoClient(MONGO_URI)
db = client[DB_NAME]

GATES = [
    ("fever", ["fever"]),
    ("resp",  ["cough","shortness_of_breath"]),
    ("gi",    ["diarrhea","vomiting"]),
    ("derm",  ["rash"]),
]

def keyword_match(sid, keywords): return any(k in sid for k in keywords)

def pick_phase1(n_total=12, include_redflags=True):
    qs = list(db.questions.find({}, {"symptom_id":1,"prompt":1,"scores":1,"is_red_flag":1}))
    red = [q for q in qs if q.get("is_red_flag")]
    non = [q for q in qs if not q.get("is_red_flag")]

    chosen, used_sids = [], set()
    for name, kws in GATES:
        gate_pool = [q for q in non if keyword_match(q["symptom_id"], kws)]
        gate_pool.sort(key=lambda q: (q["scores"]["coverage_score"] + q["scores"]["context_score"]), reverse=True)
        if gate_pool:
            chosen.append(gate_pool[0]); used_sids.add(gate_pool[0]["symptom_id"])

    # DISCRIMINATORS: highest separation_score, avoid duplicates
    remaining_slots = max(0, n_total - len(chosen))
    pool = sorted(non, key=lambda q: q["scores"]["separation_score"], reverse=True)
    for q in pool:
        if len(chosen) >= n_total: break
        if q["symptom_id"] in used_sids: continue
        chosen.append(q); used_sids.add(q["symptom_id"])

    # Optionally insert up to 2 red-flags if not already present
    if include_redflags:
        add = []
        red_sorted = sorted(red, key=lambda q: q["scores"]["context_score"], reverse=True)
        for q in red_sorted:
            if len(add) >= 2: break
            if q["symptom_id"] not in used_sids:
                add.append(q); used_sids.add(q["symptom_id"])
        # replace last items if we’re full
        for i, q in enumerate(add):
            if len(chosen) < n_total:
                chosen.append(q)
            else:
                chosen[-(i+1)] = q

    # return minimal fields UI needs
    return [
        {"symptom_id": q["symptom_id"], "prompt": q["prompt"]}
        for q in chosen[:n_total]
    ]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12, help="How many phase-1 questions")
    ap.add_argument("--no_redflags", action="store_true")
    args = ap.parse_args()

    res = pick_phase1(n_total=args.n, include_redflags=not args.no_redflags)
    print("Phase-1 set:")
    for i, q in enumerate(res, 1):
        print(f"{i:2d}. {q['prompt']}  [{q['symptom_id']}]")

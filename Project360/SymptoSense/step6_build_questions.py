from pymongo import MongoClient, ASCENDING, DESCENDING, UpdateOne
from collections import defaultdict
from statistics import mean, pstdev
from datetime import datetime

MONGO_URI = "mongodb://localhost:27017"
DB_NAME   = "symptosense"

def now(): return datetime.utcnow()

client = MongoClient(MONGO_URI)
db = client[DB_NAME]

def collect_stats():
    # symptom -> list of (weight, disease_prior)
    weights = defaultdict(list)
    df = defaultdict(int)
    total_diseases = db.diseases.estimated_document_count()

    cur = db.diseases.find({}, {
        "symptom_profile": 1,
        "priors.prevalence_score": 1
    })
    for d in cur:
        prior = float(d.get("priors", {}).get("prevalence_score", 0.5))
        seen_in_this_disease = set()
        sp = d.get("symptom_profile", {})
        for bucket in ("hallmark","core","associated","general"):
            for itm in sp.get(bucket, []) or []:
                sid = itm.get("symptom_id")
                w   = float(itm.get("weight", 0))
                if not sid: continue
                weights[sid].append((w, prior))
                if sid not in seen_in_this_disease:
                    df[sid] += 1
                    seen_in_this_disease.add(sid)

    return weights, df, total_diseases

def compute_scores(weights, df, total_diseases):
    # Turn raw stats into question scores
    # coverage_score ~ how common a symptom is across diseases
    # separation_score ~ IDF * weight_std (high = discriminative)
    # context_score ~ average disease prior where it appears
    # final question_score = 0.4*coverage + 0.4*separation + 0.2*context (this formula and values can be tweaked)
    import math

    # For coverage scaling: 75th percentile-ish ceiling
    dfs = list(df.values()) or [1]
    dfs_sorted = sorted(dfs)
    p75 = dfs_sorted[min(len(dfs_sorted)-1, int(0.75*len(dfs_sorted)))]

    out = {}
    for sid, pairs in weights.items():
        w_list  = [w for (w, _) in pairs]
        priors  = [p for (_, p) in pairs]
        avg_w   = mean(w_list) if w_list else 0.0
        std_w   = pstdev(w_list) if len(w_list) > 1 else 0.0
        df_s    = df.get(sid, 0)
        idf     = math.log((total_diseases + 1) / (df_s + 1))
        coverage_score   = min(1.0, df_s / max(1, p75))
        separation_score = idf * std_w
        context_score    = mean(priors) if priors else 0.5

        question_score   = 0.4*coverage_score + 0.4*separation_score + 0.2*context_score

        out[sid] = {
            "avg_weight": avg_w,
            "std_weight": std_w,
            "df": df_s,
            "idf": idf,
            "coverage_score": coverage_score,
            "separation_score": separation_score,
            "context_score": context_score,
            "question_score": question_score
        }
    return out

def upsert_questions(stats):
    ops = []
    for sid, s in stats.items():
        sym = db.symptoms.find_one({"_id": sid}, {"label": 1})
        label = sym["label"] if sym and sym.get("label") else sid.replace("_"," ").title()

        doc = {
            "_id": f"q_{sid}",
            "symptom_id": sid,
            "prompt": f"Do you have {label.lower()}?",
            "type": "yes_no",
            "phase": 1,            
            "scores": s,
            "is_red_flag": bool(any(kw in sid for kw in ["bleeding","chest_pain","blood_in","severe_dehydration"])),
            "updated_at": now()
        }
        ops.append(UpdateOne(
            {"_id": doc["_id"]},
            {"$setOnInsert": {"created_at": now(), "version": 1},
             "$set": doc},
            upsert=True
        ))

    if ops:
        res = db.questions.bulk_write(ops, ordered=False)
        print("questions upserted:", res.upserted_count + res.modified_count)

def create_indexes():
    db.questions.create_index([("phase", ASCENDING), ("scores.question_score", DESCENDING)], name="phase_score")
    db.questions.create_index([("is_red_flag", ASCENDING)], name="red_flag")
    db.questions.create_index([("symptom_id", ASCENDING)], name="symptom_id")

if __name__ == "__main__":
    weights, df, total = collect_stats()
    stats = compute_scores(weights, df, total)
    upsert_questions(stats)
    create_indexes()
    print("Done. Questions:", db.questions.estimated_document_count())

# step9_cases.py
from pymongo import MongoClient, UpdateOne
from datetime import datetime
import uuid

MONGO_URI = "mongodb://localhost:27017"
DB_NAME   = "symptosense"

client = MongoClient(MONGO_URI)
db = client[DB_NAME]

def utc_now(): return datetime.utcnow()

def start_case(location:str="IN", phase1_set:list[str]|None=None) -> str:
    """Create a case shell and return case_id."""
    cid = f"case_{utc_now().strftime('%Y%m%d')}_{uuid.uuid4().hex[:6]}"
    doc = {
        "_id": cid,
        "timestamp": utc_now(),
        "location": location,
        "phase1_set": phase1_set or [],      
        "answers": [],                       
        "phase2_trace": [],                 
        "candidates": [],                    
        "final_label": None,
        "feedback": None
    }
    db.cases.insert_one(doc)
    return cid

def log_phase1_answers(case_id:str, answers:dict[str,bool]):
    arr = [{"symptom_id": k, "present": bool(v)} for k,v in answers.items()]
    db.cases.update_one({"_id": case_id}, {"$addToSet": {"answers": {"$each": arr}}})

def log_phase2_step(case_id:str, asked_symptom_id:str, answer_bool:bool):
    db.cases.update_one({"_id": case_id},
                        {"$push": {"phase2_trace": {"asked": asked_symptom_id,
                                                    "answer": bool(answer_bool)}}})

def update_candidates(case_id:str, ranked:list[dict]):
    db.cases.update_one({"_id": case_id}, {"$set": {"candidates": ranked}})

def finalize_case(case_id:str, final_label:str, notes:str=""):
    case = db.cases.find_one({"_id": case_id}, {"answers":1, "phase2_trace":1})
    if not case:
        raise ValueError("case not found")

    # write final label
    db.cases.update_one({"_id": case_id},
                        {"$set": {"final_label": final_label,
                                  "feedback": {"correct": True, "notes": notes}}})

    # compile all 'present' answers across both phases
    present_ids = {a["symptom_id"] for a in case.get("answers", []) if a.get("present")}
    present_ids |= {t["asked"] for t in case.get("phase2_trace", []) if t.get("answer")}

    # increment runtime stats on the labeled disease
    incs = {"runtime_stats.cases": 1}
    for sid in present_ids:
        incs[f"runtime_stats.symptom_counts.{sid}"] = 1

    db.diseases.update_one({"_id": final_label}, {"$inc": incs})

if __name__ == "__main__":
    # quick smoke test
    cid = start_case(location="IN", phase1_set=["high_fever","rash","cough"])
    log_phase1_answers(cid, {"high_fever": True, "rash": True, "cough": False})
    log_phase2_step(cid, "retro_orbital_pain", True)
    update_candidates(cid, [{"disease_id":"dengue","score":9.2},{"disease_id":"malaria","score":6.1}])
    finalize_case(cid, "dengue", notes="Demo finalize")
    print("OK:", cid)

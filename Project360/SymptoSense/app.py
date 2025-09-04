# app.py — SymptoSense Triage (Mongo + Streamlit)
import sys
from pathlib import Path
import streamlit as st

from step9_cases import (
    start_case, log_phase1_answers, log_phase2_step,
    update_candidates, finalize_case
)

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from step7_get_phase1 import pick_phase1
from step8_next_question import next_question, score_candidates

st.set_page_config(page_title="SymptoSense — Two-Round Triage", page_icon="🩺", layout="wide")

MAX_FOLLOWUPS = 5
CONF_MARGIN = 3.5
MIN_EVIDENCE = 2.0
MIN_FUPS_BEFORE_DONE = 2

def init_state():
    ss = st.session_state
    st.session_state.setdefault("case_started_toast", False)
    ss.setdefault("stage", "phase1")
    ss.setdefault("phase1_qs", [])
    ss.setdefault("phase1_ans", {})
    ss.setdefault("asked_ids", set())
    ss.setdefault("present_ids", set())
    ss.setdefault("absent_ids", set())
    ss.setdefault("followups_asked", 0)
    ss.setdefault("last_candidates", [])
    ss.setdefault("case_id", None)
    ss.setdefault("case_started_toast", False)

init_state()

def render_candidates(title="Current candidates"):
    cands = st.session_state.last_candidates or []
    st.subheader(title)
    if not cands:
        st.info("No candidates yet. Submit answers to see results.")
        return
    for i, d in enumerate(cands, 1):
        name = d.get("name", d.get("_id", "(unnamed)"))
        score = d.get("score", 0.0)
        st.write(f"{i}. **{name}** — score: `{score:.2f}`")

def compute_and_show_candidates():
    ss = st.session_state
    cands = score_candidates(set(ss.present_ids), set(ss.absent_ids), top_k=5)
    ss.last_candidates = cands
    render_candidates()

def confident_enough():
    c = st.session_state.last_candidates or []
    if len(c) < 2:
        return False
    lead = c[0].get("score", 0.0) - c[1].get("score", 0.0)
    if lead < CONF_MARGIN:
        return False
    top = c[0]
    sp = top.get("symptom_profile", {})
    core_ids = {x["symptom_id"] for x in sp.get("core", [])} | {x["symptom_id"] for x in sp.get("hallmark", [])}
    core_w = 0.0
    for x in (sp.get("core", []) + sp.get("hallmark", [])):
        if x["symptom_id"] in st.session_state.present_ids:
            core_w += float(x.get("weight", 0))
    if core_w < MIN_EVIDENCE:
        return False
    if st.session_state.followups_asked < MIN_FUPS_BEFORE_DONE:
        return False
    return True

def reset_all():
    for k in ["stage","case_id","phase1_qs","phase1_ans","asked_ids",
              "present_ids","absent_ids","followups_asked","last_candidates","case_started_toast"]:
        if k in st.session_state:
            del st.session_state[k]
    init_state()

st.title("🩺 SymptoSense — Two-Round Triage")

with st.sidebar:
    st.markdown("### Cases")
    from pymongo import MongoClient
    client = MongoClient("mongodb://localhost:27017")
    db = client["symptosense"]

    if st.button("🔎 Show last 5 cases"):
        cases = list(
            db.cases.find({}, {"_id": 1, "timestamp": 1, "final_label": 1})
              .sort([("timestamp", -1)]).limit(5)
        )
        if not cases:
            st.info("No cases yet.")
        else:
            for c in cases:
                fl = c.get("final_label")
                if fl:
                    d = db.diseases.find_one({"_id": fl}, {"name": 1})
                    name = (d or {}).get("name", fl)
                    st.write(f"- {c['_id']} • {name}")
                else:
                    st.write(f"- {c['_id']} • Pending")



with st.sidebar:
    st.markdown("### Controls")
    if st.button("🔄 Restart"):
        reset_all()
        st.rerun()

# PHASE 1
if st.session_state.stage == "phase1":
    st.markdown("#### Phase 1 — Screening checklist")
    st.caption("We’ll ask ~12 atomic yes/no questions to narrow down the candidate diseases.")

    if not st.session_state.phase1_qs:
        st.session_state.phase1_qs = pick_phase1(n_total=12, include_redflags=True)
        if not st.session_state.case_id:
            phase1_ids = [q["symptom_id"] for q in st.session_state.phase1_qs]
            st.session_state.case_id = start_case(location="IN", phase1_set=phase1_ids)
            if not st.session_state.case_started_toast:
                st.toast(f"Case started: {st.session_state.case_id}", icon="🟢")
                st.session_state.case_started_toast = True

    with st.form("phase1_form", clear_on_submit=False):
        answers = {}
        cols = st.columns(2)
        for idx, q in enumerate(st.session_state.phase1_qs):
            sid, prompt = q["symptom_id"], q["prompt"]
            with cols[idx % 2]:
                answers[sid] = st.checkbox(prompt, key=f"p1_{sid}")
        submitted = st.form_submit_button("Submit Phase-1")
        if submitted:
            st.session_state.phase1_ans = answers
            present = {sid for sid, v in answers.items() if v}
            absent = set(answers.keys()) - present
            st.session_state.present_ids |= present
            st.session_state.absent_ids |= absent
            st.session_state.asked_ids |= set(answers.keys())
            log_phase1_answers(st.session_state.case_id, answers)
            compute_and_show_candidates()
            ranked = [
                {"disease_id": d.get("_id"), "score": float(d.get("score", 0.0))}
                for d in (st.session_state.last_candidates or [])
                if d.get("_id") is not None
            ]
            update_candidates(st.session_state.case_id, ranked)
            st.session_state.stage = "phase2"
            st.rerun()

# PHASE 2
elif st.session_state.stage == "phase2":
    st.markdown("#### Phase 2 — Follow-up (ask one at a time)")
    st.caption("We’ll pick the most informative next question from the remaining symptoms.")

    left, right = st.columns([0.6, 0.4])
    with left:
        compute_and_show_candidates()

    if confident_enough() or st.session_state.followups_asked >= MAX_FOLLOWUPS:
        st.success("We have enough information.")
        st.session_state.stage = "done"

    if st.session_state.stage == "phase2":
        best, cands, _alts = next_question(
            present_ids=st.session_state.present_ids,
            absent_ids=st.session_state.absent_ids,
            asked_ids=st.session_state.asked_ids,
            top_k=5
        )
        if not best:
            st.info("No further informative questions. Moving to finalize.")
            st.session_state.stage = "done"
        else:
            sid, prompt, _info = best
            with right:
                st.subheader("Next question")
                st.write(f"**{prompt}**")
                colA, colB = st.columns(2)
                with colA:
                    if st.button("Yes ✅", key=f"yes_{sid}"):
                        st.session_state.present_ids.add(sid)
                        st.session_state.asked_ids.add(sid)
                        st.session_state.followups_asked += 1
                        log_phase2_step(st.session_state.case_id, asked_symptom_id=sid, answer_bool=True)
                        compute_and_show_candidates()
                        ranked = [
                            {"disease_id": d.get("_id"), "score": float(d.get("score", 0.0))}
                            for d in (st.session_state.last_candidates or [])
                            if d.get("_id") is not None
                        ]
                        update_candidates(st.session_state.case_id, ranked)
                        st.rerun()
                with colB:
                    if st.button("No ❌", key=f"no_{sid}"):
                        st.session_state.absent_ids.add(sid)
                        st.session_state.asked_ids.add(sid)
                        st.session_state.followups_asked += 1
                        log_phase2_step(st.session_state.case_id, asked_symptom_id=sid, answer_bool=False)
                        compute_and_show_candidates()
                        ranked = [
                            {"disease_id": d.get("_id"), "score": float(d.get("score", 0.0))}
                            for d in (st.session_state.last_candidates or [])
                            if d.get("_id") is not None
                        ]
                        update_candidates(st.session_state.case_id, ranked)
                        st.rerun()

st.divider()
st.subheader("Finalize")

# Build options from current candidates as objects with _id + name
options = [
    {"_id": c.get("_id"), "name": c.get("name", c.get("_id", ""))}
    for c in (st.session_state.last_candidates or []) if c.get("_id")
]

final_obj = st.selectbox(
    "Pick the final diagnosis:",
    options,
    format_func=lambda o: o["name"] if o else ""
)

if st.button("Mark as Final"):
    if final_obj and final_obj.get("_id"):
        from pymongo import MongoClient
        finalize_case(st.session_state.case_id, final_obj["_id"], notes="Marked from UI")

        # VERIFY the write actually happened (defensive)
        client = MongoClient("mongodb://localhost:27017")
        db = client["symptosense"]
        saved = db.cases.find_one({"_id": st.session_state.case_id}, {"final_label": 1})

        if saved and saved.get("final_label"):
            st.session_state.stage = "done"
            st.toast(f"Saved case: {st.session_state.case_id}", icon="✅")
            st.rerun()
        else:
            st.error("Could not finalize the case. Please try again.")
    else:
        st.warning("Select a diagnosis to finalize.")

# DONE
elif st.session_state.stage == "done":
    st.markdown("#### Summary")
    st.write("**Phase-1 answers:**")
    if st.session_state.phase1_ans:
        for sid, val in st.session_state.phase1_ans.items():
            st.write(f"- {sid}: {'Yes' if val else 'No'}")

    st.write("**Phase-2 answers:**")
    phase2 = [sid for sid in st.session_state.asked_ids if sid not in st.session_state.phase1_ans]
    if phase2:
        for sid in phase2:
            v = "Yes" if sid in st.session_state.present_ids else ("No" if sid in st.session_state.absent_ids else "?")
            st.write(f"- {sid}: {v}")
    else:
        st.write("- (none)")

    render_candidates("Final candidate ranking")

    from pymongo import MongoClient
    st.success(f"✅ Case saved to MongoDB: **{st.session_state.case_id}**")

    try:
        client = MongoClient("mongodb://localhost:27017")
        db = client["symptosense"]
        case = db.cases.find_one({"_id": st.session_state.case_id}, {"final_label": 1})
        if case and case.get("final_label"):
            d = db.diseases.find_one({"_id": case["final_label"]}, {"name": 1, "runtime_stats": 1})
            if d:
                st.write(f"**Updated disease:** {d.get('name', d['_id'])}")
                rs = d.get("runtime_stats", {})
                st.write(f"- Total confirmed cases: **{rs.get('cases', 0)}**")
    except Exception as e:
        st.caption(f"(Stats preview unavailable: {e})")

    if st.button("Start a new triage"):
        reset_all()
        st.rerun()

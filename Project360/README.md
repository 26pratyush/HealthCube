# 🩺 Project360 — SymptoSense

SymptoSense is a prototype **symptom checker and disease triage tool**. It started as a **machine learning model trained on synthetic symptom–disease datasets**, but due to limited accuracy and difficulty in handling edge cases, I pivoted to a **MongoDB-backed rule/questionnaire approach**.

Instead of directly predicting diseases from free-text symptoms with ML, the current system:
1. Stores symptoms, diseases, and question flows in MongoDB.
2. Runs a **two-phase process**:
    - **Phase 1 (Checklist)**: Narrow down possible diseases.
    - **Phase 2 (Follow-up)**: Ask one-by-one clarifying questions until a confident diagnosis is reached.
3. Saves each case in MongoDB for **feedback learning** (so incorrect predictions can be corrected and reused later).

---

## 🚀 How to Run

1. Clone this repo
    ```bash
    git clone [https://github.com/](https://github.com/)<your-username>/project360-symptosense.git
    cd project360-symptosense
    ```
2. Install dependencies
    ```bash
    pip install -r requirements.txt
    ```
3. Start the app
    ```bash
    streamlit run app.py
    ```
4. Ensure MongoDB is running locally on default port 27017.

### 📂 Project Structure

* `app.py` → Main Streamlit UI for SymptoSense (connects frontend to backend logic).
* `disease_symptoms_categorization.json` → Categorized mapping of diseases and their core/associated/general symptoms.
* `Real_symptoms.txt` → Reference list of symptom names used for seeding and validation.

### Backend Pipeline (Step Scripts)

* `step1_setup.py` → Initializes MongoDB collections, sets up basic indices.
* `step2_seed_symptoms.py` → Seeds standardized symptom list into the database.
* `step3_seed_diseases.py` → Inserts diseases and their mapped symptoms into MongoDB.
* `step4_indices.py` → Builds indices for faster lookup of symptoms/diseases.
* `step5_score_candidates.py` → Logic to score and rank candidate diseases based on user-selected symptoms.
* `step6_build_questions.py` → Dynamically generates questionnaire items from symptom–disease mappings.
* `step7_get_phase1.py` → Handles initial checklist (bulk symptom screening) for Phase 1.
* `step8_next_question.py` → Runs Phase 2 follow-ups, asking one clarifying question at a time.
* `step9_cases.py` → Stores completed cases (symptoms, answers, final disease) for persistence and feedback.
* `step9_catalog_ops.py` → Helper utilities for querying and managing the stored cases catalog.

### ⚖️ Why MongoDB Instead of ML?

**ML version:** Initially, a trained model on synthetic data (XGBoost, LightGBM) could predict diseases, but:

* Accuracy dropped for overlapping symptoms (e.g., dengue vs chikungunya).
* Hard to incorporate doctor feedback directly.
* Required constant retraining with new data.

**MongoDB version:**

* Uses a knowledge-based structure instead of black-box ML.
* Easy to extend symptoms/diseases without retraining.
* Cases can be saved, corrected, and reused → improves over time.
* More explainable (every question is traceable to symptom–disease links).

### ✅ Current Status

* Phase 1 (checklist) and Phase 2 (follow-up) workflows are functional.
* MongoDB persistence is enabled: cases are saved and retrievable.
* Next planned step: feedback learning loop (reuse corrected cases to improve accuracy).

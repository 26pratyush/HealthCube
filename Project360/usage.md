# 🗄️ MongoDB Setup & Run (Local 27017)

### 1. Start MongoDB

* Install MongoDB Community Server.
* Make sure it runs on `localhost:27017`.
  Test with:

  ```bash
  mongosh
  ```

---

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 3. Initialize the database (run in order)

These scripts convert your symptom/disease files into MongoDB collections:

```bash
python step1_setup.py          # create DB + base indices
python step2_seed_symptoms.py  # load symptoms from Real_symptoms.txt
python step3_seed_diseases.py  # load diseases from JSON
python step4_indices.py        # build search indices
python step5_score_candidates.py # prepare scoring helpers
python step6_build_questions.py  # generate symptom questions
#other files will directly be used by app.py and dont need to be manually run, ex-files 7,8,9
```

After this:

* `symptoms` collection = cleaned list of all symptoms
* `diseases` collection = diseases mapped to core/associated/general symptoms
* `questions` collection = yes/no symptom questions linked to diseases
* * `cases` collection = stored history of all past triages

---

### 4. Run the app

```bash
python -m streamlit run app.py
```

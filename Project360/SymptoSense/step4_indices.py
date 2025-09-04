from pymongo import MongoClient, ASCENDING, TEXT
from step1_setup import MONGO_URI, DB_NAME

client = MongoClient(MONGO_URI)
db = client[DB_NAME]

# ---------- SYMPTOMS ----------
# Text search over labels + synonyms 
db.symptoms.create_index(
    [("label", TEXT), ("synonyms", TEXT)],
    name="symptom_text_search"
)

# Fast exact/lookups on synonyms 
db.symptoms.create_index(
    [("synonyms", ASCENDING)],
    name="idx_symptom_synonyms"
)

# ---------- DISEASES ----------
# Query-by-symptom across each bucket
db.diseases.create_index(
    [("symptom_profile.hallmark.symptom_id", ASCENDING)],
    name="idx_hallmark_sid"
)
db.diseases.create_index(
    [("symptom_profile.core.symptom_id", ASCENDING)],
    name="idx_core_sid"
)
db.diseases.create_index(
    [("symptom_profile.associated.symptom_id", ASCENDING)],
    name="idx_associated_sid"
)
db.diseases.create_index(
    [("symptom_profile.general.symptom_id", ASCENDING)],
    name="idx_general_sid"
)

# Optional: if you plan to use exclude_if frequently
db.diseases.create_index(
    [("symptom_profile.exclude_if", ASCENDING)],
    name="idx_exclude_sid"
)

# Optional: if you ever filter by prevalence
db.diseases.create_index(
    [("priors.prevalence_score", ASCENDING)],
    name="idx_prevalence"
)

# Optional: print a quick summary
print("symptoms indexes:", [ix["name"] for ix in db.symptoms.list_indexes()])
print("diseases indexes:", [ix["name"] for ix in db.diseases.list_indexes()])

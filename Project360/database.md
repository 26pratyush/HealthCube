### Overview

SymptoSense is database-driven. No global ML model. The app reads/writes these collections:

- `symptoms` — canonical symptom dictionary (+ synonyms)
- `diseases` — disease profiles (hallmark/core/associated/general buckets + weights)
- `questions` — precomputed question metadata for Phase-1 selection
- `cases` — every triage session (answers, follow-ups, final label, feedback)

# symptoms
- `_id`: canonical snake_case id (e.g. `high_fever`).
- `label`: human-readable name.
- `synonyms`: alternate phrasings.
- `value_type`: type of value (mostly `bool`).
- `units`: units if numeric (usually `null`).
- `notes`: notes string.
- `created_at` / `updated_at`: timestamps.
- `version`: doc version.
> Dictionary of atomic symptoms with synonyms.
<img width="445" height="230" alt="image" src="https://github.com/user-attachments/assets/79878a72-473e-4698-9284-ce1a13becedb" />

---

# diseases
- `_id`: canonical id (e.g. `dengue`).
- `name`: display name.
- `aliases`: alt names.
- `category_tags`: tags like “viral”.
- `symptom_profile`: buckets with `{symptom_id, weight}` for `hallmark`, `core`, `associated`, `general`, `exclude_if`.
- `runtime_stats`: `{cases, symptom_counts}` from finalized cases.
- `priors`: `{region_bias, prevalence_score}`.
- `created_at` / `updated_at`: timestamps.
- `version`: doc version.
> Semi-structured profile of each disease with weighted symptom buckets and live stats.
<img width="423" height="411" alt="image" src="https://github.com/user-attachments/assets/bcf98e55-202e-40a1-aa05-416be9b22dbf" />

---

# questions
- `_id`: `q_<symptom_id>`.
- `symptom_id`: FK to symptoms.
- `prompt`: human question text.
- `type`: question type (`yes_no`).
- `phase`: default 1.
- `is_red_flag`: safety marker.
- `scores`: `coverage`, `separation`, `context`, `question_score`.
- `created_at` / `updated_at`: timestamps.
> Precomputed metadata to choose Phase-1 questions.
<img width="421" height="394" alt="image" src="https://github.com/user-attachments/assets/97a6e542-dd2f-49b6-9644-708258df8752" />

---

# cases
- `_id`: `case_<date>_<id>`.
- `timestamp`: created time.
- `location`: region string.
- `phase1_set`: list of asked `symptom_ids`.
- `answers`: Phase-1 answers `{symptom_id, present}`.
- `phase2_trace`: follow-up Q&A.
- `candidates`: disease ranking snapshots.
- `final_label`: chosen disease id (or `null` if pending).
- `feedback`: `{correct, notes}`.
> One triage session: answers, candidates, and final diagnosis.
<img width="404" height="432" alt="image" src="https://github.com/user-attachments/assets/9384c3db-e779-40e3-ba5a-bf9f96acbd35" />

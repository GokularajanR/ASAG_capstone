# ASAG — What Was Built and How It Works

**Project:** Automated Short Answer Grading (ASAG)
**Date:** 2026-04-22

---

## What the System Does

ASAG automatically grades free-text student answers to short-answer questions. A teacher provides a question and a model (reference) answer. Students submit their own answers in natural language. The system computes how close each student's answer is to the reference using a set of NLP features, then passes those features through a trained machine learning model to produce a numeric grade on any mark scale (e.g. 0–5, 0–10).

The system is fully functional end-to-end: a teacher can log in, create a question bank, export it as a portable JSON, have students take the exam, and then grade every student across every question in a single click.

---

## Grading Logic

### How a response gets graded

Every student answer goes through the same pipeline:

1. **Preprocessing** — text is lowercased, punctuation stripped, and tokens are stemmed (so "photosynthesising" and "photosynthesis" match).
2. **Feature extraction** — six numeric features are computed by comparing the student's preprocessed response to the reference answer.
3. **GBM prediction** — the six features are fed into a trained Gradient Boosting Regressor, which outputs a grade clipped to `[0, max_marks]`.

### The six features

| # | Name | What it captures |
|---|------|-----------------|
| F1 | `sim_with_demotion` | Peer-aware similarity to the reference, with question-word demotion applied |
| F2 | `sim_no_demotion` | Same peer-aware similarity, without question-word demotion |
| F3 | `length_ratio` | `response_tokens / reference_tokens` — penalises very short or padded answers |
| F4 | `jaccard_unigram` | Stemmed unigram overlap: `|response ∩ reference| / |response ∪ reference|` |
| F5 | `keyword_coverage` | `|response ∩ reference| / |reference|` — what fraction of the reference's key terms appeared |
| F6 | `jaccard_bigram` | Bigram overlap on raw lowercase text — rewards phrase-level agreement |

### Peer-aware similarity (the key idea)

Features F1 and F2 use a **dynamic answer key** rather than a direct cosine similarity against the reference. The key is built from the entire submission batch for that question:

1. Start with term frequencies across all student responses, scaled down by a `strictness` parameter (default 20). This is the *peer signal* — terms that many students mention are given some credit even if absent from the reference.
2. **Question-word demotion** (F1 only): terms that appear in the question itself (e.g. "gravity" in "What is gravity?") are divided by 1.8 in the key. This prevents students from scoring well just by paraphrasing the question back.
3. **Reference override**: every term in the reference answer is forced to 1.0 in the key, regardless of how common it was among peers. Reference terms always carry maximum weight.

The similarity is then computed as:

```
sim = dot(response_vec, key) / sqrt(L1(response_vec) * L1(key))  × 5
```

This is a normalised dot product using L1 norms (not L2), scaled to a 0–5 range to match the scale the GBM was trained on.

### The model

A `GradientBoostingRegressor` (sklearn) with:
- 100 estimators, max depth 2, learning rate 0.1
- Subsample 0.8, min samples per leaf 10
- Trained on labelled short-answer datasets; saved as `grade_mapper.joblib`
- Output clipped to `[0, max_score]`; `max_score` is per-question and set by the teacher

Depth-2 trees are intentional: they capture feature interactions (e.g. high keyword coverage but very short response) without overfitting to training data quirks.

### Batch vs single grading

- **Single** (`main.py` CLI): grades one response; useful for debugging or inspection. Prints all six feature values and the predicted grade.
- **Batch** (API): all responses for a question are processed together. This is important because F1/F2 depend on the entire submission corpus — the peer signal changes when you add more responses. Grading happens asynchronously in a background task; the client polls for the result.

---

## What the Teacher Can Do

### Question management
- Create questions with a text, reference answer, subject tag, and maximum marks (any value, not fixed to 5).
- Delete questions from the question bank.
- **Export** the entire question bank as a portable JSON file (`question_paper.json`).
- **Import** a JSON question paper — accepts both `{ "questions": [...] }` format and bare arrays, with field aliases so externally generated files still parse correctly.

### Grading — one click for everything
- "Grade All Questions" fetches every question, retrieves all student submissions per question, runs batch grading for each, and merges the results into a complete grade summary.
- The summary is cached in the browser so all dashboard views share the same snapshot without re-running the model.
- The graded-at timestamp is displayed so the teacher knows when data was last refreshed.

### Dashboard — three views

**1. Overall results (main dashboard)**
- Stats: total students graded, mean total mark, median total mark, pass rate (≥60% of total possible marks).
- Distribution chart: students bucketed into five total-score bands (0–20%, 20–40%, 40–60%, 60–80%, 80–100%), colour-coded red to blue.
- Ranked table of all students: rank, name, roll number, total mark, marks possible, percentage. Clicking "Detail →" opens that student's individual page.
- Export all results as CSV (one row per student, with per-question grade columns).

**2. Per-question view**
- Select any question from a dropdown (or arrive via a direct link with `?q=<id>`).
- Stats specific to that question: number of responses, mean grade, median grade, pass rate.
- Distribution chart for that question's grades only.
- Table of every student's answer for that question with their grade, percentage, and the two key feature scores (peer-aware similarity and keyword coverage) shown alongside — so the teacher can see at a glance *why* a student scored what they did.

**3. Per-student view**
- Arrived at from the main table via "Detail →" link, or directly via `?roll=<roll_number>`.
- Stats: that student's total marks, marks possible, percentage, and rank out of all graded students.
- Horizontal bar chart: one bar per question showing that student's score as a percentage, colour-coded green/yellow/red. Makes it immediately obvious which topics the student is weak on.
- Full breakdown table: every question, the student's exact answer text, their grade, and percentage.
- Export that student's individual result as CSV.

---

## What the Student Can Do

- Enter name and roll number on the login page — no password; identity is used for attribution only.
- See all questions one at a time with a progress bar showing how far through the paper they are.
- Navigate freely between questions in any order using numbered dot navigation (green = answered, indigo = current, grey = not yet answered).
- Answers are auto-saved to browser storage after every keystroke — refreshing or navigating away does not lose work.
- Submit all answers at once; warned if any are unanswered before confirming.
- After submission, a confirmation screen is shown and the saved draft is cleared.

---

## Authentication

Two roles, two mechanisms:

- **Teacher**: enters an API key on the login page. The key is validated live against the server before access is granted. All write operations (create question, delete question, view submissions, start grading) require this key.
- **Student**: enters name and roll number — no server check. The roll number is used as the student identifier in the grading records and throughout all dashboard views.
- Public endpoints (list questions, submit an answer) require no authentication so the student exam works without credentials.

---

## Results Surface

After grading, the system surfaces:

- **Absolute scores**: grade out of max marks for every student × question combination.
- **Relative scores**: percentage and rank, which normalise across questions with different mark values.
- **Pass/fail split**: configurable at 60% of total marks; shown as a class-level statistic.
- **Feature transparency**: the similarity score and keyword coverage for each graded answer are visible in the per-question table — the teacher can see what the model "saw".
- **Distribution insight**: the band charts make it immediately clear whether the class performed uniformly or whether there is a bimodal split (e.g. a cluster who understood the topic and a cluster who didn't).

---

## Architecture (brief)

- **Backend**: FastAPI served by uvicorn. GBM model loaded once on startup, shared across all requests. Batch grading runs as a background task and is polled by the client.
- **Storage**: JSON files with atomic writes (write to `.tmp`, then rename). One file per collection: questions, submissions, grades, jobs. Drop-in replaceable with MongoDB or PostgreSQL later.
- **Frontend**: Vanilla HTML/CSS/JS, no framework. Three teacher pages share a grade summary cached in `localStorage` after one "Grade All" run — no extra API calls needed to navigate between views.
- **Serving**: frontend files are served directly by FastAPI's StaticFiles at `/ui/`.

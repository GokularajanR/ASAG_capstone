# Automated Short Answer Grading (ASAG)

Peer-aware automated grading for short-answer questions. Runs on a standard laptop, no GPU required.

---

## Quick start

```bash
# 1. Install dependencies
uv sync

# 2. Start the server
uv run uvicorn src.api.app:app --reload

# 3. In a second terminal, load the demo quiz
python scripts/seed_demo.py --quiz physics_quiz.json

# 4. Open the app
# http://localhost:8000/ui/
```

Log in as teacher → click **Grade All Questions** → explore the dashboard.

---

## Running the server

```bash
uv run uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

The API key is set in `.env`. Copy `.env.example` to `.env` if you haven't already.

---

## Loading the demo quiz

With the server running, in a second terminal:

```bash
python scripts/seed_demo.py --quiz physics_quiz.json
```

This loads the 10-question Middle School Physics Quiz and submits pre-written answers from three demo students (Alice Sharma, Ben Carter, Chris Patel) at good / average / weak quality levels — enough to demonstrate the full dashboard with realistic data.

Options:
```bash
python scripts/seed_demo.py --api http://localhost:8000 --key dev-key --quiz physics_quiz.json
```

---

## Grading a single response (CLI)

```bash
python main.py \
  --question "What is photosynthesis?" \
  --reference "Plants convert sunlight into glucose using chlorophyll." \
  --response "Plants use sunlight to make food."
```

Prints all six feature values and the predicted grade. Useful for debugging or inspecting individual responses.

```bash
# Options
--strictness 20     # peer key sensitivity (default 20)
--model path/to/grade_mapper.joblib
```

---

## Other scripts

| Script | What it does |
|--------|-------------|
| `python scripts/train_regressor.py --data training_data.csv` | Retrain the GBM model, saves new `grade_mapper.joblib` |
| `python scripts/evaluate.py --data training_data.csv` | Leave-one-question-out cross-validation, prints RMSE |

---

## Project structure

```
├── main.py                  # CLI entry point (single response grading)
├── physics_quiz.json        # Demo quiz (10 questions)
├── grade_mapper.joblib      # Trained GBM model
├── training_data.csv        # Labeled responses (Mohler dataset)
├── src/
│   ├── api/                 # FastAPI app, routes, auth
│   ├── features.py          # Six-feature extraction pipeline
│   ├── grade_mapper.py      # GBM wrapper
│   ├── key_builder.py       # Dynamic peer-aware key construction
│   ├── preprocessing.py     # Text normalization and stemming
│   └── store/               # JSON file storage
├── scripts/
│   ├── seed_demo.py         # Load demo quiz and student answers
│   ├── train_regressor.py   # Model training
│   └── evaluate.py          # Benchmark evaluation
├── frontend/
│   ├── student/exam.html    # Student exam interface
│   └── teacher/             # Teacher dashboard (3 views)
└── data/                    # Runtime JSON storage (Azure Files in production)
```

---

## Benchmark result

RMSE **0.81** on the Mohler dataset (2,264 responses) — better than all traditional ML baselines, no GPU required.

| System | RMSE |
|--------|------|
| TF-IDF + SVM | 1.150 |
| TF-IDF + SVR | 1.022 |
| Bag-of-Words | 0.978 |
| **This system** | **0.81** |
| RoBERTa-Large (GPU) | 0.70 |

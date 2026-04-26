"""
Train GBM grade mapper on full training dataset and save to grade_mapper.joblib.

Usage:
    python scripts/train_regressor.py [--data FILE] [--out FILE] [--strict N]
                                      [--weights] [--augment] [--embed MODEL]

Flags:
    --weights   Fix 1: inverse-frequency sample weighting (reduces high-score bias)
    --augment   Fix 6: add synthetic zero-score responses for every question
    --embed     Embedding backend: 'null' (default) or HuggingFace model name
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.evaluate import build_all_features
from src.embeddings import get_backend
from src.grade_mapper import GradeMapper

SYNTHETIC_RESPONSES = [
    "i dont know",
    "i have no idea",
    "not sure",
    "idk",
    "?",
]


def augment_with_synthetic(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, g in df.groupby("id", sort=False):
        q = g.iloc[0]
        for resp in SYNTHETIC_RESPONSES:
            rows.append({
                "id":             q["id"],
                "question":       q["question"],
                "desired_answer": q["desired_answer"],
                "student_answer": resp,
                "score_avg":      0.0,
            })
    synthetic_df = pd.DataFrame(rows)
    result = pd.concat([df, synthetic_df], ignore_index=True)
    print(f"Augmented: {len(df)} -> {len(result)} responses "
          f"(+{len(synthetic_df)} synthetic zero-score rows)")
    return result


def compute_sample_weights(scores: np.ndarray) -> np.ndarray:
    buckets = np.round(scores).astype(int).clip(0, 5)
    freq = np.bincount(buckets, minlength=6).astype(float)
    freq[freq == 0] = 1
    w = (1.0 / freq)[buckets]
    return w / w.mean()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",    default="training_data.csv")
    parser.add_argument("--out",     default="grade_mapper.joblib")
    parser.add_argument("--strict",  type=int,  default=20)
    parser.add_argument("--weights", action="store_true",
                        help="Fix 1: inverse-frequency sample weighting")
    parser.add_argument("--augment", action="store_true",
                        help="Fix 6: augment with synthetic zero-score responses")
    parser.add_argument("--embed",   default="null",
                        help="Embedding backend: 'null' or HuggingFace model name")
    args = parser.parse_args()

    backend = get_backend(args.embed)
    if not backend.is_null:
        print(f"Embedding backend: {backend.name}")

    df = pd.read_csv(args.data)
    print(f"Loaded {len(df)} responses across {df['id'].nunique()} questions.")

    if args.augment:
        df = augment_with_synthetic(df)

    X      = build_all_features(df, strictness=args.strict, embedding_backend=backend)
    scores = df["score_avg"].values

    sample_weight = compute_sample_weights(scores) if args.weights else None
    if args.weights:
        print(f"Sample weights: min={sample_weight.min():.3f}, "
              f"max={sample_weight.max():.3f}, mean={sample_weight.mean():.3f}")

    print(f"Feature matrix: {X.shape}. Training GBM...")
    mapper = GradeMapper()
    mapper.fit(X, scores, sample_weight=sample_weight)
    mapper.save(args.out)
    print(f"Model saved to {args.out}")


if __name__ == "__main__":
    main()

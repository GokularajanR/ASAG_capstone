"""
Train GBM grade mapper on full training dataset and save to grade_mapper.joblib.

Usage:
    .venv/Scripts/python.exe scripts/train_regressor.py [--data training_data.csv]
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.evaluate import build_all_features
from src.grade_mapper import GradeMapper


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",   default="training_data.csv")
    parser.add_argument("--out",    default="grade_mapper.joblib")
    parser.add_argument("--strict", type=int, default=20)
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    print(f"Loaded {len(df)} responses across {df['id'].nunique()} questions.")

    X      = build_all_features(df, strictness=args.strict)
    scores = df["score_avg"].values

    print(f"Feature matrix: {X.shape}. Training GBM...")
    mapper = GradeMapper()
    mapper.fit(X, scores)
    mapper.save(args.out)
    print(f"Model saved to {args.out}")


if __name__ == "__main__":
    main()

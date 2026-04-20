"""
Leave-one-question-out CV benchmark — GBM with 6 features.

Usage:
    .venv/Scripts/python.exe scripts/evaluate.py [--data training_data.csv]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.model_selection import LeaveOneGroupOut

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features import FEATURE_NAMES, extract_features
from src.grade_mapper import GradeMapper


def build_feature_matrix(df: pd.DataFrame, strictness: int = 20) -> np.ndarray:
    """Build (n_responses, 6) feature matrix preserving original row order."""
    blocks = []
    for _, group in df.groupby("id", sort=False):
        blocks.append(extract_features(
            question=group["question"].iloc[0],
            responses=group["student_answer"].tolist(),
            reference=group["desired_answer"].iloc[0],
            strictness=strictness,
        ))
    return np.vstack(blocks)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",   default="training_data.csv")
    parser.add_argument("--strict", type=int, default=20)
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    print(f"Dataset : {len(df)} responses, {df['id'].nunique()} questions")
    print(f"Features: {FEATURE_NAMES}")
    print("Running leave-one-question-out CV (GBM)...\n")

    X      = build_feature_matrix(df, args.strict)
    target = df["score_avg"].values
    groups = df["id"].values
    logo   = LeaveOneGroupOut()

    preds = np.zeros(len(df))
    for train_idx, test_idx in logo.split(X, target, groups):
        m = GradeMapper()
        m.fit(X[train_idx], target[train_idx])
        preds[test_idx] = m.predict_batch(X[test_idx])

    rmse    = float(np.sqrt(np.mean((preds - target) ** 2)))
    pearson = float(pearsonr(preds, target).statistic)

    print(f"RMSE      : {rmse:.4f}  (target <= 0.93)")
    print(f"Pearson r : {pearson:.4f}  (target >= 0.55)")
    print(f"\n{'[PASS]' if rmse <= 0.93 else '[FAIL]'} RMSE")
    print(f"{'[PASS]' if pearson >= 0.55 else '[FAIL]'} Pearson r")

    # Feature importances from a model trained on all data
    full_model = GradeMapper()
    full_model.fit(X, target)
    print("\nFeature importances (full-data model):")
    for name, imp in sorted(zip(FEATURE_NAMES, full_model.feature_importances_),
                             key=lambda x: -x[1]):
        print(f"  {name:<25} {imp:.4f}")


# expose for train_regressor
def build_all_features(df: pd.DataFrame, strictness: int = 20) -> np.ndarray:
    return build_feature_matrix(df, strictness)


if __name__ == "__main__":
    main()

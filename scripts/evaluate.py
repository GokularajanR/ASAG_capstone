"""
Leave-one-question-out CV benchmark.

Usage:
    python scripts/evaluate.py [--data FILE] [--strict N] [--nof7] [--embed MODEL]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.model_selection import LeaveOneGroupOut

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embeddings import get_backend
from src.features import FEATURE_NAMES, extract_features
from src.grade_mapper import GradeMapper


def build_feature_matrix(
    df: pd.DataFrame,
    strictness: int = 20,
    use_f7: bool = True,
    embedding_backend=None,
) -> np.ndarray:
    """Build feature matrix preserving original row order."""
    blocks = []
    for _, group in df.groupby("id", sort=False):
        blocks.append(extract_features(
            question=group["question"].iloc[0],
            responses=group["student_answer"].tolist(),
            reference=group["desired_answer"].iloc[0],
            strictness=strictness,
            use_f7=use_f7,
            embedding_backend=embedding_backend,
        ))
    return np.vstack(blocks)


def _active_feature_names(use_f7: bool, use_f8: bool) -> list[str]:
    names = FEATURE_NAMES[:6]
    if use_f7:
        names = names + [FEATURE_NAMES[6]]
    if use_f8:
        names = names + [FEATURE_NAMES[7]]
    return names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",   default="training_data.csv")
    parser.add_argument("--strict", type=int, default=20)
    parser.add_argument("--nof7",   action="store_true",
                        help="Use only 6 features (baseline-compatible)")
    parser.add_argument("--embed",  default="null",
                        help="Embedding backend: 'null' or HuggingFace model name")
    args = parser.parse_args()

    use_f7  = not args.nof7
    backend = get_backend(args.embed)
    use_f8  = not backend.is_null
    feature_names = _active_feature_names(use_f7, use_f8)

    df = pd.read_csv(args.data)
    print(f"Dataset : {len(df)} responses, {df['id'].nunique()} questions")
    print(f"Features: {feature_names}")
    if use_f8:
        print(f"Embed   : {backend.name}")
    print("Running leave-one-question-out CV (GBM)...\n")

    X      = build_feature_matrix(df, args.strict, use_f7=use_f7, embedding_backend=backend)
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

    full_model = GradeMapper()
    full_model.fit(X, target)
    print("\nFeature importances (full-data model):")
    for name, imp in sorted(zip(feature_names, full_model.feature_importances_),
                             key=lambda x: -x[1]):
        print(f"  {name:<25} {imp:.4f}")


# expose for train_regressor and compare_checkpoints
def build_all_features(
    df: pd.DataFrame,
    strictness: int = 20,
    use_f7: bool = True,
    embedding_backend=None,
) -> np.ndarray:
    return build_feature_matrix(df, strictness, use_f7=use_f7, embedding_backend=embedding_backend)


if __name__ == "__main__":
    main()

"""
Compare model configurations via leave-one-question-out CV.

  v0 — baseline:           6 features, no weights, no augmentation
  v1 — Fix 1 + Fix 5:      7 features, sample weights
  v2 — Fix 1 + 5 + 6:      7 features, sample weights, synthetic augmentation
  v3 — Fix 1 + 5 + embed:  8 features, sample weights, embedding angle (all-MiniLM-L6-v2)

Usage:
    python scripts/compare_checkpoints.py [--data FILE] [--strict N] [--skip-embed]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.model_selection import LeaveOneGroupOut

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.evaluate import build_all_features
from scripts.train_regressor import augment_with_synthetic, compute_sample_weights
from src.embeddings import NullBackend, get_backend
from src.grade_mapper import GradeMapper

CONFIGS = [
    {"name": "v0  baseline",            "use_f7": False, "weights": False, "augment": False, "embed": "null"},
    {"name": "v1  fix1+fix5",           "use_f7": True,  "weights": True,  "augment": False, "embed": "null"},
    {"name": "v2  fix1+fix5+fix6",      "use_f7": True,  "weights": True,  "augment": True,  "embed": "null"},
    {"name": "v3  fix1+fix5+embed",     "use_f7": True,  "weights": True,  "augment": False, "embed": "all-MiniLM-L6-v2"},
]


def run_logo_cv(df, strictness, use_f7, use_weights, backend):
    X      = build_all_features(df, strictness=strictness, use_f7=use_f7, embedding_backend=backend)
    target = df["score_avg"].values
    groups = df["id"].values
    logo   = LeaveOneGroupOut()

    preds = np.zeros(len(df))
    for train_idx, test_idx in logo.split(X, target, groups):
        m  = GradeMapper()
        sw = compute_sample_weights(target[train_idx]) if use_weights else None
        m.fit(X[train_idx], target[train_idx], sample_weight=sw)
        preds[test_idx] = m.predict_batch(X[test_idx])
    return preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",       default="training_data.csv")
    parser.add_argument("--strict",     type=int, default=20)
    parser.add_argument("--skip-embed", action="store_true",
                        help="Skip v3 (embedding) to save time")
    args = parser.parse_args()

    base_df   = pd.read_csv(args.data)
    zero_mask = base_df["score_avg"] == 0.0
    print(f"Dataset: {len(base_df)} responses, {base_df['id'].nunique()} questions\n")

    results = []
    for cfg in CONFIGS:
        if args.skip_embed and cfg["embed"] != "null":
            print(f"Skipping {cfg['name']} (--skip-embed)")
            continue

        print(f"Running LOGO CV for {cfg['name']} ...")
        df      = augment_with_synthetic(base_df) if cfg["augment"] else base_df.copy()
        backend = get_backend(cfg["embed"])

        preds  = run_logo_cv(df, args.strict, cfg["use_f7"], cfg["weights"], backend)
        target = df["score_avg"].values

        rmse    = float(np.sqrt(np.mean((preds - target) ** 2)))
        pearson = float(pearsonr(preds, target).statistic)
        mean_on_zero = float(np.mean(preds[:len(base_df)][zero_mask.values]))

        results.append({
            "config":       cfg["name"],
            "embed":        cfg["embed"] if cfg["embed"] != "null" else "-",
            "rmse":         rmse,
            "pearson":      pearson,
            "mean_pred":    float(np.mean(preds)),
            "mean_on_zero": mean_on_zero,
        })
        print(f"  RMSE={rmse:.4f}  Pearson={pearson:.4f}  "
              f"mean_pred={float(np.mean(preds)):.3f}  mean_on_zero={mean_on_zero:.3f}\n")

    print("=" * 100)
    print(f"{'Config':<28} {'Embed':<22} {'RMSE':>7} {'Pearson':>8} {'MeanPred':>9} {'MeanOnZero':>11}")
    print("-" * 100)
    for r in results:
        print(f"{r['config']:<28} {r['embed']:<22} {r['rmse']:>7.4f} "
              f"{r['pearson']:>8.4f} {r['mean_pred']:>9.3f} {r['mean_on_zero']:>11.3f}")
    print("=" * 100)
    print("\nKey columns:")
    print("  RMSE        -- lower is better overall")
    print("  Pearson     -- higher is better correlation with human grades")
    print("  MeanOnZero  -- mean predicted grade on true-zero responses (lower = less inflation)")


if __name__ == "__main__":
    main()

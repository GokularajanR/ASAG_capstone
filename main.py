"""
ASAG CLI — grade a single student response from the command line.

Usage:
    .venv/Scripts/python.exe main.py \\
        --question "What is photosynthesis?" \\
        --reference "Plants convert light into glucose." \\
        --response  "Plants use sunlight to make food." \\
        [--corpus response1.txt response2.txt ...] \\
        [--model grade_mapper.joblib] \\
        [--strictness 20]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.features import FEATURE_NAMES, extract_features
from src.grade_mapper import GradeMapper


def grade_single(
    question: str,
    reference: str,
    response: str,
    corpus: list[str] | None = None,
    model_path: str | Path = "grade_mapper.joblib",
    strictness: int = 20,
) -> dict:
    """
    Grade one student response using the 6-feature GBM pipeline.

    corpus  : other student responses for the same question (peer signal).
              If None, the single response is used as its own corpus.
    """
    corpus = corpus or [response]

    # response is always the first element so we can index row 0
    all_responses = [response] + [r for r in corpus if r != response]
    feat_matrix   = extract_features(question, all_responses, reference, strictness)
    feat_vec      = feat_matrix[0]  # features for the target response

    mapper = GradeMapper.load(model_path)
    grade  = mapper.predict(feat_vec)

    return {
        "features": dict(zip(FEATURE_NAMES, feat_vec.round(4).tolist())),
        "predicted_grade": round(grade, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="ASAG — grade a student response")
    parser.add_argument("--question",   required=True, help="Question text")
    parser.add_argument("--reference",  required=True, help="Model/reference answer")
    parser.add_argument("--response",   required=True, help="Student response to grade")
    parser.add_argument("--corpus",     nargs="*",     help="Paths to text files with other responses")
    parser.add_argument("--model",      default="grade_mapper.joblib", help="Path to saved GradeMapper")
    parser.add_argument("--strictness", type=int, default=20)
    args = parser.parse_args()

    corpus: list[str] = []
    if args.corpus:
        for path in args.corpus:
            corpus.append(Path(path).read_text(encoding="utf-8"))

    result = grade_single(
        question=args.question,
        reference=args.reference,
        response=args.response,
        corpus=corpus or None,
        model_path=args.model,
        strictness=args.strictness,
    )

    print("Features:")
    for k, v in result["features"].items():
        print(f"  {k:<25} {v}")
    print(f"\nPredicted grade  : {result['predicted_grade']} / 5.0")


if __name__ == "__main__":
    main()

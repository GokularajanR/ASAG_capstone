"""POST /grade — synchronous single-response grading."""

from fastapi import APIRouter, Depends

from src.api.auth import require_api_key
from src.api.deps import get_mapper
from src.api.models import GradeRequest, GradeResponse
from src.features import FEATURE_NAMES, extract_features

router = APIRouter(dependencies=[Depends(require_api_key)])


@router.post("/grade", response_model=GradeResponse)
def grade(req: GradeRequest, mapper=Depends(get_mapper)):
    corpus = req.corpus or [req.response]
    all_responses = [req.response] + [r for r in corpus if r != req.response]
    feat_matrix = extract_features(req.question, all_responses, req.reference, req.strictness)
    feat_vec = feat_matrix[0]
    grade_val = mapper.predict(feat_vec)
    return GradeResponse(
        predicted_grade=round(grade_val, 2),
        features={k: round(float(v), 4) for k, v in zip(FEATURE_NAMES, feat_vec)},
    )

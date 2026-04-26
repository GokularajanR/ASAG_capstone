"""POST /grade — synchronous single-response grading."""

from fastapi import APIRouter, Depends

from src.api.auth import require_api_key
from src.api.deps import get_embedding_backend, get_mapper
from src.api.models import GradeRequest, GradeResponse
from src.features import FEATURE_NAMES, extract_features
from src.grade_mapper import snap_grade

router = APIRouter(dependencies=[Depends(require_api_key)])


@router.post("/grade", response_model=GradeResponse)
def grade(
    req: GradeRequest,
    mapper=Depends(get_mapper),
    embedding_backend=Depends(get_embedding_backend),
):
    corpus = req.corpus or [req.response]
    all_responses = [req.response] + [r for r in corpus if r != req.response]
    feat_matrix = extract_features(
        req.question, all_responses, req.reference, req.strictness,
        embedding_backend=embedding_backend,
    )
    feat_vec  = feat_matrix[0]
    n_feats   = len(feat_vec)
    raw       = mapper.predict(feat_vec)
    grade_val = snap_grade(min(raw * (req.max_marks / 5.0), req.max_marks))
    return GradeResponse(
        predicted_grade=grade_val,
        features={k: round(float(v), 4) for k, v in zip(FEATURE_NAMES[:n_feats], feat_vec)},
    )

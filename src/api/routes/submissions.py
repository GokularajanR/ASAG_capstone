"""POST /submissions (public), GET /submissions (teacher auth)."""

from fastapi import APIRouter, Depends, Query

from src.api.auth import require_api_key
from src.api.models import SubmissionIn, SubmissionOut
from src.store.collections import SubmissionStore

router = APIRouter(prefix="/submissions")

_store = SubmissionStore()


@router.post("", response_model=SubmissionOut, status_code=201)
def submit_answer(body: SubmissionIn):
    from datetime import datetime, timezone
    data = body.model_dump()
    data["submitted_at"] = datetime.now(timezone.utc).isoformat()
    record = _store.insert(data)
    return record


@router.get("", response_model=list[SubmissionOut], dependencies=[Depends(require_api_key)])
def list_submissions(question_id: str = Query(default="")):
    return _store.find_by_question(question_id) if question_id else _store.all()

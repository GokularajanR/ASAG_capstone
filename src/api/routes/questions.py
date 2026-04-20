"""GET|POST /questions, GET /questions/{id}"""

from fastapi import APIRouter, Depends, HTTPException

from src.api.auth import require_api_key
from src.api.deps import get_question_store
from src.api.models import QuestionIn, QuestionOut
from src.store.collections import QuestionStore

router = APIRouter(prefix="/questions", dependencies=[Depends(require_api_key)])


@router.post("", response_model=QuestionOut, status_code=201)
def create_question(body: QuestionIn, store: QuestionStore = Depends(get_question_store)):
    record = store.insert(body.model_dump())
    return record


@router.get("", response_model=list[QuestionOut])
def list_questions(subject: str = "", store: QuestionStore = Depends(get_question_store)):
    return store.find_by_subject(subject) if subject else store.all()


@router.get("/{question_id}", response_model=QuestionOut)
def get_question(question_id: str, store: QuestionStore = Depends(get_question_store)):
    record = store.get(question_id)
    if not record:
        raise HTTPException(status_code=404, detail="Question not found.")
    return record

"""POST /questions/{id}/batch, GET /jobs/{id}"""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from src.api.auth import require_api_key
from src.api.deps import get_embedding_backend, get_grade_store, get_job_store, get_mapper, get_question_store
from src.api.models import BatchRequest, JobOut
from src.features import FEATURE_NAMES, extract_features
from src.grade_mapper import snap_grade
from src.store.collections import GradeStore, JobStore, QuestionStore

router = APIRouter(dependencies=[Depends(require_api_key)])


def _run_batch(
    job_id: str,
    question_text: str,
    reference: str,
    responses: list[str],
    student_ids: list[str],
    strictness: int,
    max_marks: float,
    mapper,
    embedding_backend,
    job_store: JobStore,
    grade_store: GradeStore,
) -> None:
    try:
        job_store.update(job_id, {"status": "running"})
        feat_matrix = extract_features(
            question_text, responses, reference, strictness,
            embedding_backend=embedding_backend,
        )
        raw_grades = mapper.predict_batch(feat_matrix)
        grades = [snap_grade(min(g * (max_marks / 5.0), max_marks)) for g in raw_grades]
        n_feats = feat_matrix.shape[1]
        for i, grade_val in enumerate(grades):
            grade_store.insert({
                "job_id":          job_id,
                "response_id":     student_ids[i] if i < len(student_ids) else str(i),
                "predicted_grade": grade_val,
                "features":        {k: round(float(v), 4)
                                    for k, v in zip(FEATURE_NAMES[:n_feats], feat_matrix[i])},
                "model_version":   "1.0",
            })
        job_store.mark_done(job_id)
    except Exception as exc:
        job_store.mark_done(job_id, error=str(exc))


@router.post("/questions/{question_id}/batch", response_model=JobOut, status_code=202)
def submit_batch(
    question_id: str,
    body: BatchRequest,
    background_tasks: BackgroundTasks,
    mapper=Depends(get_mapper),
    embedding_backend=Depends(get_embedding_backend),
    q_store: QuestionStore = Depends(get_question_store),
    job_store: JobStore = Depends(get_job_store),
    grade_store: GradeStore = Depends(get_grade_store),
):
    question = q_store.get(question_id)
    if not question:
        raise HTTPException(status_code=404, detail="Question not found.")
    if not body.responses:
        raise HTTPException(status_code=422, detail="responses list is empty.")

    job = job_store.insert({
        "question_id": question_id,
        "status":      "pending",
        "strictness":  body.strictness,
    })
    background_tasks.add_task(
        _run_batch,
        job["id"], question["text"], question["reference_answer"],
        body.responses, body.student_ids, body.strictness,
        question.get("max_marks", 5.0),
        mapper, embedding_backend, job_store, grade_store,
    )
    return job


@router.get("/jobs/{job_id}", response_model=JobOut)
def get_job(
    job_id: str,
    job_store: JobStore = Depends(get_job_store),
    grade_store: GradeStore = Depends(get_grade_store),
):
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job["status"] == "done":
        job["grades"] = grade_store.find_by_job(job_id)
    return job

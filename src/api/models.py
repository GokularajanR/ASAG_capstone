"""Pydantic request and response models for the ASAG API."""

from __future__ import annotations

from pydantic import BaseModel, Field


# ------------------------------------------------------------------ requests

class GradeRequest(BaseModel):
    question: str
    reference: str
    response: str
    corpus: list[str] = Field(default_factory=list)
    strictness: int = 20


class QuestionIn(BaseModel):
    text: str
    reference_answer: str
    subject: str = ""


class BatchRequest(BaseModel):
    responses: list[str]
    student_ids: list[str] = Field(default_factory=list)
    strictness: int = 20


class UserIn(BaseModel):
    email: str
    role: str = "student"
    password: str


# ------------------------------------------------------------------ responses

class GradeResponse(BaseModel):
    predicted_grade: float
    features: dict[str, float]


class QuestionOut(BaseModel):
    id: str
    text: str
    reference_answer: str
    subject: str
    created_at: str


class JobOut(BaseModel):
    id: str
    status: str
    created_at: str
    completed_at: str | None = None
    error: str | None = None
    grades: list[dict] | None = None


class UserOut(BaseModel):
    id: str
    email: str
    role: str
    created_at: str

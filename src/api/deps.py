"""Shared dependency factories for stores and the grade mapper."""

from __future__ import annotations

from fastapi import Request

from src.store.collections import (
    GradeStore, JobStore, QuestionStore, ResponseStore, UserStore,
)


def get_mapper(request: Request):
    return request.app.state.mapper


def get_question_store() -> QuestionStore:
    return QuestionStore()


def get_response_store() -> ResponseStore:
    return ResponseStore()


def get_grade_store() -> GradeStore:
    return GradeStore()


def get_job_store() -> JobStore:
    return JobStore()


def get_user_store() -> UserStore:
    return UserStore()

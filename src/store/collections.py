"""Typed store subclasses — one per JSON collection."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from .base import JsonStore

_DATA = Path("data")


class QuestionStore(JsonStore):
    def __init__(self, path: Path | str = _DATA / "questions.json") -> None:
        super().__init__(path)

    def find_by_subject(self, subject: str) -> list[dict]:
        return self.find(subject=subject)


class ResponseStore(JsonStore):
    def __init__(self, path: Path | str = _DATA / "responses.json") -> None:
        super().__init__(path)

    def find_by_question(self, question_id: str) -> list[dict]:
        return self.find(question_id=question_id)


class GradeStore(JsonStore):
    def __init__(self, path: Path | str = _DATA / "grades.json") -> None:
        super().__init__(path)

    def find_by_response(self, response_id: str) -> list[dict]:
        return self.find(response_id=response_id)

    def find_by_job(self, job_id: str) -> list[dict]:
        return self.find(job_id=job_id)


class JobStore(JsonStore):
    def __init__(self, path: Path | str = _DATA / "jobs.json") -> None:
        super().__init__(path)

    def find_by_status(self, status: str) -> list[dict]:
        return self.find(status=status)

    def mark_done(self, id: str, error: str | None = None) -> dict:
        patch = {
            "status": "failed" if error else "done",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "error": error,
        }
        return self.update(id, patch)


class UserStore(JsonStore):
    def __init__(self, path: Path | str = _DATA / "users.json") -> None:
        super().__init__(path)

    def find_by_email(self, email: str) -> dict | None:
        results = self.find(email=email)
        return results[0] if results else None

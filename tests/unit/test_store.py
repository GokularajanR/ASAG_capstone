"""Unit tests for JSON file-based store layer."""

import pytest
from src.store.base import JsonStore
from src.store.collections import (
    QuestionStore, ResponseStore, GradeStore, JobStore, UserStore
)


# ------------------------------------------------------------------ fixtures

@pytest.fixture
def store(tmp_path):
    return JsonStore(tmp_path / "test.json")


@pytest.fixture
def q_store(tmp_path):
    return QuestionStore(tmp_path / "questions.json")


@pytest.fixture
def r_store(tmp_path):
    return ResponseStore(tmp_path / "responses.json")


@pytest.fixture
def g_store(tmp_path):
    return GradeStore(tmp_path / "grades.json")


@pytest.fixture
def j_store(tmp_path):
    return JobStore(tmp_path / "jobs.json")


@pytest.fixture
def u_store(tmp_path):
    return UserStore(tmp_path / "users.json")


# ------------------------------------------------------------------ base CRUD

def test_insert_and_get(store):
    rec = store.insert({"name": "alice"})
    assert rec["id"]
    assert store.get(rec["id"]) == rec


def test_insert_auto_id(store):
    r1 = store.insert({"x": 1})
    r2 = store.insert({"x": 2})
    assert r1["id"] != r2["id"]


def test_all(store):
    store.insert({"a": 1})
    store.insert({"a": 2})
    assert len(store.all()) == 2


def test_find_filter(store):
    store.insert({"role": "teacher"})
    store.insert({"role": "student"})
    store.insert({"role": "student"})
    assert len(store.find(role="student")) == 2
    assert len(store.find(role="teacher")) == 1


def test_update_patches_fields(store):
    rec = store.insert({"x": 1, "y": 2})
    updated = store.update(rec["id"], {"x": 99})
    assert updated["x"] == 99
    assert updated["y"] == 2


def test_update_missing_raises(store):
    with pytest.raises(KeyError):
        store.update("nonexistent", {"x": 1})


def test_delete_returns_true(store):
    rec = store.insert({"z": 0})
    assert store.delete(rec["id"]) is True
    assert store.get(rec["id"]) is None


def test_delete_missing_returns_false(store):
    assert store.delete("nonexistent") is False


def test_atomic_write_no_tmp_left(store):
    store.insert({"k": "v"})
    tmp = store._path.with_suffix(".tmp")
    assert not tmp.exists()


def test_persistence(tmp_path):
    path = tmp_path / "persist.json"
    s1 = JsonStore(path)
    rec = s1.insert({"val": 42})

    s2 = JsonStore(path)
    assert s2.get(rec["id"])["val"] == 42


def test_empty_store_returns_empty(store):
    assert store.all() == []
    assert store.get("any") is None


# ------------------------------------------------------------------ typed collections

def test_question_find_by_subject(q_store):
    q_store.insert({"text": "Q1", "reference_answer": "A1", "subject": "bio"})
    q_store.insert({"text": "Q2", "reference_answer": "A2", "subject": "chem"})
    results = q_store.find_by_subject("bio")
    assert len(results) == 1
    assert results[0]["text"] == "Q1"


def test_response_find_by_question(r_store):
    r_store.insert({"question_id": "q1", "text": "ans1", "student_id": "s1"})
    r_store.insert({"question_id": "q2", "text": "ans2", "student_id": "s2"})
    assert len(r_store.find_by_question("q1")) == 1


def test_grade_find_by_response(g_store):
    g_store.insert({"response_id": "r1", "predicted_grade": 3.5, "features": {}})
    g_store.insert({"response_id": "r2", "predicted_grade": 4.0, "features": {}})
    assert len(g_store.find_by_response("r1")) == 1


def test_job_find_by_status(j_store):
    j_store.insert({"question_id": "q1", "status": "pending"})
    j_store.insert({"question_id": "q2", "status": "done"})
    j_store.insert({"question_id": "q3", "status": "pending"})
    assert len(j_store.find_by_status("pending")) == 2


def test_job_mark_done(j_store):
    job = j_store.insert({"question_id": "q1", "status": "running"})
    updated = j_store.mark_done(job["id"])
    assert updated["status"] == "done"
    assert updated["completed_at"] is not None
    assert updated["error"] is None


def test_job_mark_done_with_error(j_store):
    job = j_store.insert({"question_id": "q1", "status": "running"})
    updated = j_store.mark_done(job["id"], error="timeout")
    assert updated["status"] == "failed"
    assert updated["error"] == "timeout"


def test_user_find_by_email(u_store):
    u_store.insert({"email": "a@test.com", "role": "teacher", "hashed_password": "x"})
    u_store.insert({"email": "b@test.com", "role": "student", "hashed_password": "y"})
    result = u_store.find_by_email("a@test.com")
    assert result is not None
    assert result["role"] == "teacher"


def test_user_find_by_email_missing(u_store):
    assert u_store.find_by_email("nobody@test.com") is None

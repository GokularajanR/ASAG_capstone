"""FastAPI application — ASAG grading API."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

load_dotenv()

from src.grade_mapper import GradeMapper
from src.api.routes import grade, jobs, questions, users

MODEL_PATH = Path(os.getenv("MODEL_PATH", "grade_mapper.joblib"))
FRONTEND_DIR = (Path(__file__).parent.parent.parent / "frontend").resolve()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.mapper = GradeMapper.load(MODEL_PATH)
    yield


app = FastAPI(title="ASAG API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(grade.router)
app.include_router(questions.router)
app.include_router(jobs.router)
app.include_router(users.router)


@app.get("/health")
def health():
    return {"status": "ok"}


if FRONTEND_DIR.exists():
    app.mount("/ui", StaticFiles(directory=FRONTEND_DIR, html=True), name="ui")

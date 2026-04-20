"""POST /users — create a user with hashed password."""

import hashlib

from fastapi import APIRouter, Depends, HTTPException

from src.api.auth import require_api_key
from src.api.deps import get_user_store
from src.api.models import UserIn, UserOut
from src.store.collections import UserStore

router = APIRouter(prefix="/users", dependencies=[Depends(require_api_key)])


def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


@router.post("", response_model=UserOut, status_code=201)
def create_user(body: UserIn, store: UserStore = Depends(get_user_store)):
    if store.find_by_email(body.email):
        raise HTTPException(status_code=409, detail="Email already registered.")
    record = store.insert({
        "email": body.email,
        "role": body.role,
        "hashed_password": _hash_password(body.password),
    })
    return record

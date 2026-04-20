"""Atomic JSON file store — base class for all collections."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path


class JsonStore:
    """
    Stores records as a JSON object keyed by id.
    Writes are atomic: data is flushed to a .tmp file then renamed.
    """

    _path: Path

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ I/O

    def _load(self) -> dict:
        if not self._path.exists():
            return {}
        return json.loads(self._path.read_text(encoding="utf-8"))

    def _save(self, data: dict) -> None:
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self._path)

    # ------------------------------------------------------------------ CRUD

    def all(self) -> list[dict]:
        return list(self._load().values())

    def get(self, id: str) -> dict | None:
        return self._load().get(id)

    def insert(self, record: dict) -> dict:
        data = self._load()
        if not record.get("id"):
            record = {**record, "id": str(uuid.uuid4())}
        if "created_at" not in record:
            record["created_at"] = datetime.now(timezone.utc).isoformat()
        data[record["id"]] = record
        self._save(data)
        return record

    def update(self, id: str, patch: dict) -> dict:
        data = self._load()
        if id not in data:
            raise KeyError(f"Record {id!r} not found.")
        data[id] = {**data[id], **patch, "id": id}
        self._save(data)
        return data[id]

    def delete(self, id: str) -> bool:
        data = self._load()
        if id not in data:
            return False
        del data[id]
        self._save(data)
        return True

    def find(self, **kwargs) -> list[dict]:
        return [r for r in self._load().values()
                if all(r.get(k) == v for k, v in kwargs.items())]

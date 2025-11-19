from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional
from uuid import uuid4


class AttendanceLogBook:
    """Persist attendance sessions as named logs with book-style navigation."""

    def __init__(self, path: Path, *, max_entries: int = 500) -> None:
        self.path = path
        self.max_entries = max_entries
        self.pages: list[dict[str, object]] = []
        self.selected_page_id: Optional[str] = None
        self._entry_cache: dict[str, list[dict[str, object]]] = {}
        self._load()

    def _load(self) -> None:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            payload = {}
        except (OSError, json.JSONDecodeError):
            payload = {}
        raw_pages = payload.get("pages") if isinstance(payload, dict) else []
        normalized: list[dict[str, object]] = []
        if isinstance(raw_pages, list):
            for page in raw_pages:
                if isinstance(page, dict):
                    normalized_page = self._normalize_page(page)
                    normalized.append(normalized_page)
                    entries = page.get("entries")
                    if isinstance(entries, list):
                        page_id = normalized_page["id"]
                        cached = [
                            self._normalize_entry(entry) for entry in entries[: self.max_entries]
                        ]
                        self._entry_cache[page_id] = cached
        self.pages = normalized
        selected = payload.get("selected_page_id") if isinstance(payload, dict) else None
        if selected and self.get_page(selected):
            self.selected_page_id = selected
        elif self.pages:
            self.selected_page_id = self.pages[0]["id"]
        else:
            self.selected_page_id = None

    def _save(self) -> None:
        pages_payload: list[dict[str, object]] = []
        for page in self.pages:
            page_id = page["id"]
            entries = self._entry_cache.get(page_id)
            if entries is None:
                entries = self._load_entries_from_disk(page_id)
            payload_page = dict(page)
            payload_page["entries"] = entries[: self.max_entries] if entries else []
            pages_payload.append(payload_page)
        payload = {
            "version": 1,
            "selected_page_id": self.selected_page_id,
            "pages": pages_payload,
        }
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self.path.with_name(self.path.name + ".tmp")
            tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            tmp_path.replace(self.path)
        except OSError:
            pass

    def _normalize_page(self, raw: dict[str, object]) -> dict[str, object]:
        created = str(raw.get("created_at") or datetime.now(timezone.utc).isoformat(timespec="seconds"))
        entries_raw = raw.get("entries")
        entry_count = len(entries_raw) if isinstance(entries_raw, list) else 0
        return {
            "id": str(raw.get("id") or uuid4().hex),
            "name": str(raw.get("name") or "Session"),
            "created_at": created,
            "updated_at": str(raw.get("updated_at") or created),
            "entry_count": entry_count,
        }

    def _normalize_entry(self, entry: dict[str, object]) -> dict[str, object]:
        normalized = dict(entry)
        normalized["timestamp"] = str(entry.get("timestamp") or "")
        normalized["identity"] = str(entry.get("identity") or "Unknown")
        normalized["accepted"] = bool(entry.get("accepted"))
        normalized["source"] = str(entry.get("source") or "")
        return normalized

    @property
    def is_empty(self) -> bool:
        return not self.pages

    def list_pages(self) -> list[dict[str, object]]:
        return list(self.pages)

    def get_page(self, page_id: Optional[str]) -> Optional[dict[str, object]]:
        if not page_id:
            return None
        for page in self.pages:
            if page["id"] == page_id:
                return page
        return None

    def entries_for(self, page_id: Optional[str], *, limit: Optional[int] = None) -> list[dict[str, object]]:
        if not page_id:
            return []
        entries = self._get_cached_entries(page_id)
        if limit is not None:
            return list(entries[:limit])
        return list(entries)

    def page_index(self, page_id: Optional[str]) -> int:
        if not page_id:
            return -1
        for idx, page in enumerate(self.pages):
            if page["id"] == page_id:
                return idx
        return -1

    def create_page(self, name: str) -> dict[str, object]:
        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        page = {
            "id": uuid4().hex,
            "name": name or "Session",
            "created_at": timestamp,
            "updated_at": timestamp,
            "entry_count": 0,
        }
        self.pages.append(page)
        self._entry_cache[page["id"]] = []
        self._save()
        return page

    def rename_page(self, page_id: str, new_name: str) -> bool:
        page = self.get_page(page_id)
        if not page:
            return False
        page["name"] = new_name or page["name"]
        page["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self._save()
        return True

    def delete_page(self, page_id: str) -> bool:
        page = self.get_page(page_id)
        if page is None:
            return False
        self.pages = [p for p in self.pages if p["id"] != page_id]
        self._entry_cache.pop(page_id, None)
        if self.selected_page_id == page_id:
            self.selected_page_id = self.pages[0]["id"] if self.pages else None
        self._save()
        return True

    def add_entry(self, page_id: str, entry: dict[str, object]) -> None:
        normalized = self._normalize_entry(entry)
        entries = self._get_cached_entries(page_id)
        entries.insert(0, normalized)
        entries[:] = entries[: self.max_entries]
        page = self.get_page(page_id)
        if page:
            page["entry_count"] = len(entries)
            page["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            self._save()

    def _get_cached_entries(self, page_id: str) -> list[dict[str, object]]:
        if page_id not in self._entry_cache:
            self._entry_cache[page_id] = self._load_entries_from_disk(page_id)
        return self._entry_cache[page_id]

    def _load_entries_from_disk(self, page_id: str) -> list[dict[str, object]]:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return []
        pages = payload.get("pages") if isinstance(payload, dict) else []
        if isinstance(pages, list):
            for page in pages:
                if not isinstance(page, dict):
                    continue
                if str(page.get("id")) != page_id:
                    continue
                entries = page.get("entries")
                if isinstance(entries, list):
                    return [
                        self._normalize_entry(entry)
                        for entry in entries[: self.max_entries]
                        if isinstance(entry, dict)
                    ]
        return []

    def set_selected_page(self, page_id: Optional[str]) -> None:
        if page_id and self.get_page(page_id):
            self.selected_page_id = page_id
            self._save()

    def release_entries(self, keep: Optional[Iterable[str]] = None) -> None:
        keep_set = set(keep or [])
        for key in list(self._entry_cache.keys()):
            if key not in keep_set:
                self._entry_cache.pop(key, None)


__all__ = ["AttendanceLogBook"]

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import datetime
from typing import Optional

import tkinter as tk
from tkinter import ttk

from dashboard.utils import _format_display_timestamp


class AttendanceLog(ttk.Frame):
    def __init__(self, master: tk.Misc) -> None:
        super().__init__(master)
        columns = ("timestamp", "identity", "result", "source")
        self._columns = columns
        self._heading_labels = {
            "timestamp": "Timestamp",
            "identity": "Identity",
            "result": "Result",
            "source": "Source",
        }
        self._sort_column: Optional[str] = None
        self._sort_descending = False

        self.tree = ttk.Treeview(self, columns=columns, show="headings", height=10)
        for col in columns:
            self.tree.heading(col, text=self._heading_labels[col], command=lambda c=col: self._sort_tree(c))
            self.tree.column(col, anchor="w")
        vsb = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

    def add_entry(self, *, timestamp: str, identity: str, accepted: bool, source: str, reapply_sort: bool = True) -> None:
        result = "Accepted" if accepted else "Rejected"
        display_time = _format_display_timestamp(timestamp)
        self.tree.insert("", 0, values=(display_time, identity, result, source))
        self._trim_rows()
        if reapply_sort:
            self._reapply_sort_if_needed()

    def _trim_rows(self) -> None:
        children = self.tree.get_children()
        for item in children[200:]:
            self.tree.delete(item)

    def clear(self) -> None:
        for item in self.tree.get_children():
            self.tree.delete(item)

    def load_entries(self, entries: Iterable[dict[str, object]]) -> None:
        self.clear()
        sequence: Sequence[dict[str, object]]
        if isinstance(entries, Sequence):
            sequence = entries
        else:
            sequence = list(entries)
        for entry in reversed(sequence):
            self.add_entry(
                timestamp=str(entry.get("timestamp", "")),
                identity=str(entry.get("identity", "")),
                accepted=bool(entry.get("accepted")),
                source=str(entry.get("source", "")),
                reapply_sort=False,
            )
        self._reapply_sort_if_needed()

    def _reapply_sort_if_needed(self) -> None:
        if self._sort_column:
            self._sort_tree(self._sort_column, toggle=False, descending=self._sort_descending)

    def _sort_tree(self, column: str, *, toggle: bool = True, descending: Optional[bool] = None) -> None:
        items = self.tree.get_children("")
        if not items:
            return
        if descending is None:
            if toggle and self._sort_column == column:
                descending = not self._sort_descending
            elif toggle:
                descending = False
            else:
                descending = self._sort_descending if self._sort_column == column else False
        sort_payload: list[tuple[tuple[int, object], str]] = []
        for item in items:
            value = self.tree.set(item, column)
            sort_payload.append((self._sort_key(column, value), item))
        sort_payload.sort(key=lambda entry: entry[0], reverse=bool(descending))
        for index, (_, item) in enumerate(sort_payload):
            self.tree.move(item, "", index)
        self._sort_column = column
        self._sort_descending = bool(descending)
        self._update_heading_labels()

    def _sort_key(self, column: str, raw_value: str) -> tuple[int, object]:
        text = (raw_value or "").strip()
        if column == "timestamp":
            parsed = self._parse_display_timestamp(text)
            if parsed is not None:
                return (0, parsed)
            return (1, text.lower())
        if column == "result":
            order = 0 if text.lower().startswith("accept") else 1
            return (order, text.lower())
        return (0, text.lower())

    @staticmethod
    def _parse_display_timestamp(value: str) -> Optional[datetime]:
        if not value:
            return None
        for fmt in ("%b %d, %Y %I:%M %p", "%b %d, %Y %I:%M%p"):
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                continue
        return None

    def _update_heading_labels(self) -> None:
        for col in self._columns:
            label = self._heading_labels[col]
            if col == self._sort_column:
                arrow = "▼" if self._sort_descending else "▲"
                label = f"{label} {arrow}"
            self.tree.heading(col, text=label)

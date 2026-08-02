"""
data_loader.py

Loads all dataset CSVs and builds fast lookup indexes so that, for any
incoming message, we can pull every piece of context relevant to the
routing decision: the receiving user's behavior, the group/sender
relationship, the business's trust signals, and the user's history
with that business.

Nothing here calls an LLM. This is pure, deterministic data plumbing,
which keeps the pipeline fast, free, and easy to debug.
"""

from __future__ import annotations

import csv
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional


def _read_csv(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@dataclass
class Dataset:
    dataset_dir: str

    messages: list[dict] = field(default_factory=list)
    sample_messages: list[dict] = field(default_factory=list)
    users: dict[str, dict] = field(default_factory=dict)
    groups: dict[str, dict] = field(default_factory=dict)
    # (group_id, user_id) -> membership row
    group_members: dict[tuple[str, str], dict] = field(default_factory=dict)
    businesses: dict[str, dict] = field(default_factory=dict)
    # (user_id, business_id) -> history row
    user_business_history: dict[tuple[str, str], dict] = field(default_factory=dict)
    message_history: dict[str, dict] = field(default_factory=dict)  # message_id -> row
    # user_id -> list of history rows (their inbox history), most recent last
    message_history_by_user: dict[str, list[dict]] = field(default_factory=lambda: defaultdict(list))
    # message_id -> event row (assumes 1 event per historical message; falls back to list if needed)
    message_events: dict[str, dict] = field(default_factory=dict)
    images: dict[str, dict] = field(default_factory=dict)  # image_id -> row
    voice_notes: dict[str, dict] = field(default_factory=dict)  # voice_note_id -> row
    daily_summary_by_user: dict[str, list[dict]] = field(default_factory=lambda: defaultdict(list))

    @classmethod
    def load(cls, dataset_dir: str) -> "Dataset":
        ds = cls(dataset_dir=dataset_dir)

        def p(name: str) -> str:
            return os.path.join(dataset_dir, name)

        ds.messages = _read_csv(p("messages.csv"))

        if os.path.exists(p("sample_messages.csv")):
            ds.sample_messages = _read_csv(p("sample_messages.csv"))

        for row in _read_csv(p("users.csv")):
            ds.users[row["user_id"]] = row

        for row in _read_csv(p("groups.csv")):
            ds.groups[row["group_id"]] = row

        for row in _read_csv(p("group_members.csv")):
            ds.group_members[(row["group_id"], row["user_id"])] = row

        for row in _read_csv(p("business_accounts.csv")):
            ds.businesses[row["business_id"]] = row

        for row in _read_csv(p("user_business_history.csv")):
            ds.user_business_history[(row["user_id"], row["business_id"])] = row

        for row in _read_csv(p("message_history.csv")):
            ds.message_history[row["message_id"]] = row
            ds.message_history_by_user[row["user_id"]].append(row)

        # Sort each user's history chronologically so "most recent" is meaningful.
        for uid in ds.message_history_by_user:
            ds.message_history_by_user[uid].sort(key=lambda r: r.get("created_at", ""))

        for row in _read_csv(p("message_events.csv")):
            # If duplicates exist, last one wins; in practice this dataset has 1:1.
            ds.message_events[row["message_id"]] = row

        for row in _read_csv(p("images.csv")):
            ds.images[row["image_id"]] = row

        for row in _read_csv(p("voice_notes.csv")):
            ds.voice_notes[row["voice_note_id"]] = row

        if os.path.exists(p("daily_notification_summary.csv")):
            for row in _read_csv(p("daily_notification_summary.csv")):
                ds.daily_summary_by_user[row["user_id"]].append(row)

        return ds

    # ---- convenience accessors -------------------------------------------------

    def user(self, user_id: str) -> Optional[dict]:
        return self.users.get(user_id)

    def group(self, group_id: Optional[str]) -> Optional[dict]:
        if not group_id:
            return None
        return self.groups.get(group_id)

    def membership(self, group_id: Optional[str], user_id: str) -> Optional[dict]:
        if not group_id:
            return None
        return self.group_members.get((group_id, user_id))

    def business(self, business_id: Optional[str]) -> Optional[dict]:
        if not business_id:
            return None
        return self.businesses.get(business_id)

    def biz_history(self, user_id: str, business_id: Optional[str]) -> Optional[dict]:
        if not business_id:
            return None
        return self.user_business_history.get((user_id, business_id))

    def event_for(self, message_id: str) -> Optional[dict]:
        return self.message_events.get(message_id)

    def media_path(self, media_type: str, media_id: str) -> Optional[str]:
        if not media_id:
            return None
        if media_type == "image":
            row = self.images.get(media_id)
        elif media_type == "voice":
            row = self.voice_notes.get(media_id)
        else:
            return None
        if not row:
            return None
        rel = row.get("file_path")
        if not rel:
            return None
        return os.path.join(self.dataset_dir, rel)

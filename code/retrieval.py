"""
retrieval.py

Finds historical messages relevant to an incoming message, for the same
user, and attaches how the user actually reacted to them (opened,
replied, dismissed, muted, reported). This is what lets the router say
things like "the user has ignored 3 similar messages before" instead of
guessing.

Approach: lightweight TF-IDF-style cosine similarity over message text,
restricted to the same user's history (and boosted for same sender /
same group / same business), no external ML dependencies needed. This
keeps the whole pipeline free and fast, and the LLM only ever sees a
short, pre-filtered candidate list rather than the full 1000+ row history.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from data_loader import Dataset

_WORD_RE = re.compile(r"[a-z0-9]+")

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "to", "of", "and",
    "in", "on", "for", "with", "this", "that", "it", "your", "you", "our",
    "we", "at", "by", "from", "as", "or", "will", "can", "has", "have",
    "please", "pls", "just", "now", "if", "so", "but", "not", "no",
}


def _tokenize(text: str) -> list[str]:
    if not text:
        return []
    return [w for w in _WORD_RE.findall(text.lower()) if w not in _STOPWORDS and len(w) > 2]


@dataclass
class Evidence:
    message_id: str
    score: float
    row: dict
    event: dict | None


class Retriever:
    """Per-dataset retriever. Builds a simple IDF table once, reuses it for every query."""

    def __init__(self, ds: Dataset):
        self.ds = ds
        self._doc_freq: Counter[str] = Counter()
        self._n_docs = 0
        self._tokens_cache: dict[str, list[str]] = {}

        for mid, row in ds.message_history.items():
            toks = _tokenize(row.get("message_text", ""))
            self._tokens_cache[mid] = toks
            if toks:
                self._n_docs += 1
                for w in set(toks):
                    self._doc_freq[w] += 1

    def _idf(self, word: str) -> float:
        df = self._doc_freq.get(word, 0)
        return math.log((self._n_docs + 1) / (df + 1)) + 1.0

    def _vec(self, tokens: list[str]) -> Counter[str]:
        tf = Counter(tokens)
        return Counter({w: c * self._idf(w) for w, c in tf.items()})

    @staticmethod
    def _cosine(a: Counter[str], b: Counter[str]) -> float:
        if not a or not b:
            return 0.0
        dot = sum(a[w] * b.get(w, 0.0) for w in a)
        na = math.sqrt(sum(v * v for v in a.values()))
        nb = math.sqrt(sum(v * v for v in b.values()))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    def top_evidence(
        self,
        *,
        user_id: str,
        message_text: str,
        sender_user_id: str | None,
        group_id: str | None,
        business_id: str | None,
        media_type: str | None,
        top_k: int = 3,
        min_score: float = 0.12,
    ) -> list[Evidence]:
        """
        Returns up to top_k historical messages belonging to this user,
        ranked primarily by RELATIONSHIP signals -- same sender, same
        business, same group, and how strongly the user reacted to those
        past messages (opened/replied/reported vs. ignored/dismissed) --
        with text similarity used as a smaller refinement on top, mainly
        to disambiguate between several messages from the same
        sender/business/group.

        Rationale: "has this exact sender/business contacted this user
        before, and what did they do about it" is a much stronger and
        more literally relevant signal for evidence_message_ids than word
        overlap. Two messages can be near-identical in wording but from
        totally unrelated senders (a generic "reminder" template used by
        many businesses); relationship match should dominate. Text
        similarity still matters as a tiebreaker -- e.g. picking the most
        similar of several same-sender messages -- and remains the only
        signal available when there is no sender/business/group to match
        on (e.g. bare personal messages with a new sender).
        """
        candidates = self.ds.message_history_by_user.get(user_id, [])
        if not candidates:
            return []

        query_tokens = _tokenize(message_text or "")
        query_vec = self._vec(query_tokens) if query_tokens else Counter()

        scored: list[Evidence] = []
        for row in candidates:
            mid = row["message_id"]
            event = self.ds.event_for(mid)

            relationship_score = 0.0
            same_sender = bool(sender_user_id and row.get("sender_user_id") == sender_user_id)
            same_business = bool(business_id and row.get("business_id") == business_id)
            same_group = bool(group_id and row.get("group_id") == group_id)
            same_media = bool(media_type and row.get("media_type") == media_type and media_type != "")

            # Relationship match is the dominant signal. Business and
            # direct-sender repetition are the strongest "this user has a
            # history with this exact source" indicators; group co-
            # membership is weaker on its own since groups have many
            # senders, so it counts for less unless paired with same_sender.
            if same_sender:
                relationship_score += 0.55
            if same_business:
                relationship_score += 0.55
            if same_group:
                relationship_score += 0.20
            if same_media:
                relationship_score += 0.05

            # Reaction-history boost: a past message from this exact
            # sender/business is even more informative as evidence when we
            # know how the user reacted to it (either direction -- a clear
            # "always ignores this" or "always engages with this" pattern
            # is exactly the kind of evidence that should surface).
            if (same_sender or same_business) and event:
                if event.get("message_reported") == "1":
                    relationship_score += 0.25  # strong safety-relevant evidence
                elif event.get("muted_after_message") == "1":
                    relationship_score += 0.15
                elif event.get("message_replied") == "1" or event.get("message_opened") == "1":
                    relationship_score += 0.10
                elif event.get("notification_dismissed") == "1":
                    relationship_score += 0.10

            # Text similarity: a smaller refinement, mainly to rank between
            # several candidates that already share a relationship match,
            # and the only signal at all when there's no sender/business/
            # group overlap to key off (e.g. a first-time personal sender).
            text_score = 0.0
            if query_vec:
                doc_vec = self._vec(self._tokens_cache.get(mid, []))
                text_score = self._cosine(query_vec, doc_vec)

            if relationship_score > 0:
                # Relationship-anchored: text similarity only nudges the
                # ranking, it doesn't need to carry the score on its own.
                score = relationship_score + (text_score * 0.25)
            else:
                # No relationship signal at all -- fall back to pure text
                # similarity so we can still surface a genuinely similar
                # past message (e.g. same scam template from a new sender).
                score = text_score

            if score >= min_score:
                scored.append(Evidence(message_id=mid, score=score, row=row, event=event))

        scored.sort(key=lambda e: e.score, reverse=True)
        return scored[:top_k]

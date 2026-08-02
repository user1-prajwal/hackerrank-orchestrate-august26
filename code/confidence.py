"""
confidence.py

Computes routing confidence primarily from deterministic features rather
than trusting an LLM's self-reported number, which tends to cluster
around similar values regardless of how strong the actual evidence is.

Signals used:
  - retrieval strength: how strong the top evidence match was (relationship
    + reaction history, from retrieval.py's own scoring)
  - sender/business trust: verified business, domain match, known vs.
    first-contact relationship
  - user interaction history: how much reliable signal exists for this
    user/sender/business pair (more history = more confidence either way)
  - safety score: presence/absence of risk signals from safety_rules.py

Gemini's own confidence is only used as a fallback when none of the
deterministic signals are strong enough to produce a confident number on
their own -- i.e. for genuinely ambiguous cases where the model's
judgment is the only real signal available.
"""

from __future__ import annotations

from dataclasses import dataclass

from data_loader import Dataset
from retrieval import Evidence
from safety_rules import check_text_risk_signals, domain_mismatch


@dataclass
class ConfidenceBreakdown:
    value: float
    basis: str  # short label for why, useful for debugging/reason text
    deterministic: bool  # False if we had to fall back to the LLM's own confidence


def _retrieval_strength(evidence: list[Evidence]) -> float:
    if not evidence:
        return 0.0
    top = evidence[0].score
    # Evidence scores from retrieval.py can exceed 1.0 (relationship +
    # reaction boosts stack); clamp into a 0-1 "how strong is our best
    # match" signal.
    return max(0.0, min(1.0, top))


def _trust_strength(ds: Dataset, msg: dict) -> float:
    """How well-established / verifiable is this sender relationship, 0-1."""
    conv_type = msg.get("conversation_type")
    business_id = msg.get("business_id") or None
    group_id = msg.get("group_id") or None
    user_id = msg["user_id"]
    sender_user_id = msg.get("sender_user_id") or None

    if conv_type == "business" and business_id:
        biz = ds.business(business_id)
        hist = ds.biz_history(user_id, business_id)
        score = 0.0
        if biz and biz.get("verified") == "1":
            score += 0.4
        if biz and not domain_mismatch(ds, business_id):
            score += 0.2
        if hist:
            score += 0.4
        return min(1.0, score)

    if conv_type == "group" and group_id:
        member = ds.membership(group_id, user_id)
        sender_member = ds.membership(group_id, sender_user_id) if sender_user_id else None
        score = 0.3  # baseline: group membership itself is some signal
        if member:
            score += 0.2
        if sender_member and sender_member.get("role") == "admin":
            score += 0.3
        return min(1.0, score)

    if conv_type == "personal":
        # Personal senders have no separate trust table; treat as moderate
        # baseline trust, refined by history strength elsewhere.
        return 0.5

    return 0.3


def _history_volume_strength(evidence: list[Evidence]) -> float:
    """More corroborating history (up to a point) = more confidence in the pattern."""
    n = len(evidence)
    if n == 0:
        return 0.0
    if n == 1:
        return 0.4
    if n == 2:
        return 0.7
    return 1.0


def _safety_strength(msg: dict) -> tuple[float, bool]:
    """
    Returns (strength, is_risky). strength is how confidently we can say
    "this looks safe" (high) or "this looks risky" (also high, just in
    the other direction) -- i.e. how far from ambiguous the safety
    signals are. is_risky flags which direction.
    """
    text = msg.get("message_text") or ""
    signals = check_text_risk_signals(text)
    n_signals = sum(signals.values())
    if n_signals == 0:
        return 0.7, False  # no risk signals at all: fairly confident it's benign
    if n_signals == 1:
        return 0.4, True  # one weak signal: ambiguous
    return 0.85, True  # multiple risk signals: confidently risky


def compute_confidence(
    ds: Dataset,
    msg: dict,
    evidence: list[Evidence],
    llm_confidence: float | None,
) -> ConfidenceBreakdown:
    """
    Blends deterministic signals into a single confidence value. Falls
    back toward the LLM's self-reported confidence only when the
    deterministic signals are individually weak (i.e. genuinely
    ambiguous case), and even then blends rather than fully overriding,
    so a wildly overconfident LLM number can't dominate.
    """
    retrieval_strength = _retrieval_strength(evidence)
    trust_strength = _trust_strength(ds, msg)
    history_strength = _history_volume_strength(evidence)
    safety_strength, is_risky = _safety_strength(msg)

    # Weighted blend of deterministic signals. Safety gets the highest
    # weight since a confident risk read should dominate a confident
    # "this sender is verified" read when they disagree.
    deterministic_value = (
        0.30 * retrieval_strength
        + 0.20 * trust_strength
        + 0.15 * history_strength
        + 0.35 * safety_strength
    )
    deterministic_value = max(0.0, min(0.97, deterministic_value))

    # How much do we trust this deterministic estimate on its own? If
    # every component is weak/uninformative (new sender, no history, no
    # clear safety read either way), the deterministic number is really
    # just noise around a prior -- that's when we lean on the LLM.
    signal_strength = max(retrieval_strength, trust_strength - 0.3, history_strength, abs(safety_strength - 0.55))

    if llm_confidence is None:
        return ConfidenceBreakdown(value=round(deterministic_value, 2), basis="deterministic_only", deterministic=True)

    if signal_strength >= 0.35:
        # Deterministic signal is meaningful: use it as primary, let the
        # LLM's number nudge it only slightly.
        blended = 0.8 * deterministic_value + 0.2 * llm_confidence
        return ConfidenceBreakdown(value=round(max(0.0, min(0.97, blended)), 2), basis="deterministic_primary", deterministic=True)

    # Deterministic signals are all weak/ambiguous -- genuinely ambiguous
    # case, so lean on the LLM's judgment, but still keep a light anchor
    # to the deterministic estimate so it can't run away to extremes.
    blended = 0.35 * deterministic_value + 0.65 * llm_confidence
    return ConfidenceBreakdown(value=round(max(0.0, min(0.97, blended)), 2), basis="llm_fallback", deterministic=False)

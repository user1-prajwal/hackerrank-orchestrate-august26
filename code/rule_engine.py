"""
rule_engine.py

A lightweight, deterministic rule layer that runs AFTER safety_rules.py
(which handles hard scam overrides) and BEFORE the Gemini call. Its job
is to resolve the "easy" majority of messages without spending an LLM
call, so Gemini is reserved for:

  - messages with an image or voice note (multimodal reasoning is the
    whole reason to use an LLM there), and
  - messages that are genuinely ambiguous once the deterministic signals
    are weighed.

This matters for two reasons: it keeps the pipeline well inside free-tier
daily quotas, and it makes the easy cases MORE consistent, not less --
a clean rule firing the same way every time beats an LLM call with
sampling variance on something that was never actually ambiguous.

Design: each rule looks at retrieval + relationship signals already
computed elsewhere (group mute state, business verification/opt-out,
repeated-ignore patterns, admin/direct-mention status) and only fires
when the evidence is strong. If no rule fires, the message is marked
"needs_llm" and falls through to Gemini exactly as before.

Nothing here is scam-specific (that stays in safety_rules.py). This
module handles routine, high-confidence notify/digest/mute calls that
don't need risk judgment -- e.g. a muted group with a routine update,
a business the user has clearly opted out of, or a direct verified-
business message matching a live order.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from data_loader import Dataset
from retrieval import Evidence

_GREETING_FORWARD = re.compile(
    r"\b(good morning|good night|forwarded as received|fwd as received|share (this |it )?with (everyone|all)|"
    r"stay positive|keep smiling|good morning all)\b",
    re.I,
)

_DIRECT_ASK = re.compile(
    r"\b(can you|could you|please (reply|confirm|call|send)|@you|are you (free|available)|"
    r"let me know|need your (help|input|confirmation))\b",
    re.I,
)

_PAYMENT_URGENCY_HINT = re.compile(
    r"\b(pay|payment|charge|fee|bank|account|verify|verification|otp|pin|password|link|"
    r"urgent(ly)?|expire|expiry|block(ed)?|suspend(ed)?|clearance|scan and pay|"
    r"benefit approval|token|deposit)\b",
    re.I,
)

_NEGATED_URGENCY = re.compile(r"\b(nothing|no|not)\s+urgent(ly)?\b", re.I)


def _has_payment_urgency_hint(text: str) -> bool:
    """
    True if the text contains payment/urgency-adjacent language that
    warrants deferring to the LLM instead of a blunt rule-based mute.
    Explicitly discounts common negated phrasing ("nothing urgent", "no
    urgency") so that reassuring, low-stakes chatter isn't bounced to the
    LLM just for saying the opposite of what we're looking for.
    """
    stripped = _NEGATED_URGENCY.sub("", text)
    return bool(_PAYMENT_URGENCY_HINT.search(stripped))

# How many prior interactions we require before trusting a "user always
# ignores this" or "user always engages with this" pattern.
_MIN_HISTORY_FOR_PATTERN = 2


@dataclass
class RuleVerdict:
    resolved: bool
    action: str | None = None
    message_type: str | None = None
    reason: str | None = None
    confidence: float | None = None
    needs_llm: bool = True  # only meaningful when resolved=False


def _reaction_summary(evidence: list[Evidence]) -> dict:
    """Summarizes how the user reacted to a list of same-sender/business/group evidence."""
    opened = replied = dismissed = muted = reported = 0
    n = len(evidence)
    for ev in evidence:
        e = ev.event or {}
        opened += e.get("message_opened") == "1"
        replied += e.get("message_replied") == "1"
        dismissed += e.get("notification_dismissed") == "1"
        muted += e.get("muted_after_message") == "1"
        reported += e.get("message_reported") == "1"
    return {"n": n, "opened": opened, "replied": replied, "dismissed": dismissed,
            "muted": muted, "reported": reported}


def evaluate_rules(
    ds: Dataset,
    msg: dict,
    evidence: list[Evidence],
) -> RuleVerdict:
    """
    Attempts to resolve the routing decision deterministically. Returns
    resolved=False (needs_llm=True) if no rule is confident enough to fire,
    which is the common/expected case for genuinely ambiguous text-only
    messages -- those still go to Gemini exactly as before.

    Media messages (image/voice) always fall through to Gemini regardless
    of what rules match, since reading the attachment requires the LLM.
    """
    media_type = msg.get("media_type") or ""
    if media_type in ("image", "voice"):
        return RuleVerdict(resolved=False, needs_llm=True)

    text = msg.get("message_text") or ""
    conv_type = msg.get("conversation_type")
    user_id = msg["user_id"]
    group_id = msg.get("group_id") or None
    business_id = msg.get("business_id") or None
    sender_user_id = msg.get("sender_user_id") or None
    forwarded_count = int(msg.get("forwarded_count") or 0)

    same_sender_ev = [e for e in evidence if sender_user_id and e.row.get("sender_user_id") == sender_user_id]
    same_business_ev = [e for e in evidence if business_id and e.row.get("business_id") == business_id]
    same_group_ev = [e for e in evidence if group_id and e.row.get("group_id") == group_id]

    # ---- Rule 1: muted group + routine chatter, no direct ask ----------
    if conv_type == "group" and group_id:
        member = ds.membership(group_id, user_id)
        if member and member.get("group_muted_by_user") == "1":
            is_direct_ask = bool(_DIRECT_ASK.search(text))
            mentions_user = f"@{user_id}" in text
            has_payment_urgency_hint = _has_payment_urgency_hint(text)

            if not is_direct_ask and not mentions_user:
                if has_payment_urgency_hint:
                    # This could be a scam riding on a muted group rather
                    # than genuinely routine chatter -- the blunt "muted =
                    # mute, type unknown" call isn't precise enough here.
                    # Defer to the LLM (and safety_rules.py already ran
                    # before this and would have caught business-domain
                    # phishing patterns) rather than mislabeling it.
                    return RuleVerdict(resolved=False, needs_llm=True)
                return RuleVerdict(
                    resolved=True,
                    action="mute",
                    message_type="unknown",
                    reason="The user has muted this group and the message is routine chatter with no direct ask or mention.",
                    confidence=0.8,
                    needs_llm=False,
                )

    # ---- Rule 2: business opted out of promotions -----------------------
    if conv_type == "business" and business_id:
        hist = ds.biz_history(user_id, business_id)
        if hist and hist.get("promotions_opted_out_at"):
            return RuleVerdict(
                resolved=True,
                action="mute",
                message_type="promotion",
                reason="The user has explicitly opted out of promotional messages from this business.",
                confidence=0.85,
                needs_llm=False,
            )
        if hist and hist.get("allows_promotions") == "0" and same_business_ev:
            summary = _reaction_summary(same_business_ev)
            if summary["n"] >= _MIN_HISTORY_FOR_PATTERN and summary["opened"] == 0 and summary["replied"] == 0:
                return RuleVerdict(
                    resolved=True,
                    action="mute",
                    message_type="promotion",
                    reason="The user does not allow promotions from this business and has consistently ignored similar past messages.",
                    confidence=0.78,
                    needs_llm=False,
                )

    # ---- Rule 3: greeting/forward chain with repeat-ignore pattern ------
    if _GREETING_FORWARD.search(text) or forwarded_count >= 3:
        pattern_ev = same_sender_ev or same_group_ev
        summary = _reaction_summary(pattern_ev)
        if summary["n"] >= _MIN_HISTORY_FOR_PATTERN and summary["opened"] == 0 and summary["replied"] == 0:
            return RuleVerdict(
                resolved=True,
                action="mute",
                message_type="forward",
                reason="This looks like a forwarded greeting/chain message, and the user has ignored similar messages from this sender before.",
                confidence=0.75,
                needs_llm=False,
            )
        if forwarded_count >= 3 and not pattern_ev:
            # High forward count alone, no history either way: safe default
            # to digest rather than confidently muting or notifying.
            return RuleVerdict(
                resolved=True,
                action="digest",
                message_type="forward",
                reason="The message has been forwarded multiple times, which typically indicates low-priority broadcast content.",
                confidence=0.6,
                needs_llm=False,
            )

    # ---- Rule 4: verified business, user has live/recent relationship,
    #              message matches an ongoing order/booking pattern -------
    if conv_type == "business" and business_id:
        biz = ds.business(business_id)
        hist = ds.biz_history(user_id, business_id)
        if biz and biz.get("verified") == "1" and hist:
            why = (hist.get("why_user_knows_account") or "").lower()
            has_recent_activity = bool(hist.get("last_activity_at"))
            is_operational_word = bool(re.search(
                r"\b(delivery|order|booking|appointment|reminder|confirm(ed)?|arriving|scheduled|pickup|otp)\b",
                text, re.I,
            ))
            # Only fire when there's no domain mismatch and no risk-signal
            # overlap -- safety_rules.py already caught the risky cases
            # before this module runs, but we stay conservative here too.
            official = (biz.get("official_domain") or "").strip().lower()
            used = (biz.get("domain_used_by_sender") or "").strip().lower()
            domain_ok = not official or not used or official == used
            if has_recent_activity and is_operational_word and domain_ok and why in (
                "recent_grocery_delivery", "recent_order", "recent_booking", "active_booking",
            ):
                return RuleVerdict(
                    resolved=True,
                    action="notify",
                    message_type="business_update" if "order" in text.lower() or "deliver" in text.lower() else "event",
                    reason="A verified business with a live, recent order/booking relationship sent an operational update matching that context.",
                    confidence=0.82,
                    needs_llm=False,
                )

    # ---- Rule 5: direct personal message with an explicit ask -----------
    if conv_type == "personal" and _DIRECT_ASK.search(text):
        return RuleVerdict(
            resolved=True,
            action="notify",
            message_type="personal",
            reason="This is a direct personal message where the sender explicitly asks the user for a response or action.",
            confidence=0.8,
            needs_llm=False,
        )

    # No rule confidently resolved this message -- defer to Gemini, which
    # has richer context (and, for media, is required anyway).
    return RuleVerdict(resolved=False, needs_llm=True)

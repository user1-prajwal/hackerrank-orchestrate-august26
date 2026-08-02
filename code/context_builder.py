"""
context_builder.py

Turns raw joined rows into a compact, human-readable context block that
we hand to the LLM. Keeping this readable (rather than raw JSON dumps)
makes the model's reasoning noticeably better and the prompt shorter,
which matters for free-tier rate limits.
"""

from __future__ import annotations

from dataclasses import dataclass

from data_loader import Dataset
from retrieval import Evidence, Retriever


def _fmt_event(ev: dict | None) -> str:
    if not ev:
        return "no recorded reaction"
    parts = []
    if ev.get("message_opened") == "1":
        parts.append("opened")
    if ev.get("message_replied") == "1":
        parts.append("replied")
    if ev.get("notification_dismissed") == "1":
        parts.append("dismissed")
    if ev.get("muted_after_message") == "1":
        parts.append("muted sender/group after this")
    if ev.get("message_reported") == "1":
        parts.append("REPORTED")
    if not parts:
        parts.append("ignored (no open/reply)")
    rt = ev.get("reaction_time_minutes")
    if rt:
        parts.append(f"reacted in {rt}m")
    return ", ".join(parts)


@dataclass
class MessageContext:
    message_id: str
    conversation_type: str
    message_text: str
    media_type: str
    media_path: str | None
    forwarded_count: int
    prompt_context_block: str
    evidence: list[Evidence]


def build_context(ds: Dataset, retriever: Retriever, msg: dict) -> MessageContext:
    user_id = msg["user_id"]
    conversation_type = msg["conversation_type"]
    group_id = msg.get("group_id") or None
    business_id = msg.get("business_id") or None
    sender_user_id = msg.get("sender_user_id") or None
    media_type = msg.get("media_type") or ""
    media_id = msg.get("media_id") or None
    message_text = msg.get("message_text") or ""
    forwarded_count = int(msg.get("forwarded_count") or 0)

    lines: list[str] = []

    # --- receiving user behavior ---
    user = ds.user(user_id)
    lines.append("## Receiving user")
    if user:
        lines.append(
            f"- user_id={user_id}, quiet_hours={user.get('do_not_disturb_window','?')}, "
            f"opened_30d={user.get('messages_opened_30d','?')}, "
            f"replied_30d={user.get('messages_replied_30d','?')}, "
            f"dismissed_30d={user.get('notifications_dismissed_30d','?')}, "
            f"reported_30d={user.get('messages_reported_30d','?')}"
        )
        try:
            opened = int(user.get("messages_opened_30d") or 0)
            dismissed = int(user.get("notifications_dismissed_30d") or 0)
            reported = int(user.get("messages_reported_30d") or 0)
            if reported and reported >= 2:
                lines.append("  -> This user reports suspicious messages fairly often: weigh safety signals heavily.")
            if dismissed > opened:
                lines.append("  -> This user dismisses more than they open: they are generally low-tolerance for noise.")
        except ValueError:
            pass
    else:
        lines.append(f"- user_id={user_id} (no profile found)")

    # --- conversation-specific context ---
    if conversation_type == "group" and group_id:
        group = ds.group(group_id)
        member = ds.membership(group_id, user_id)
        lines.append("\n## Group context")
        if group:
            lines.append(
                f"- group='{group.get('group_name')}', type={group.get('group_type')}, "
                f"members={group.get('member_count')}, admins={group.get('admin_count')}, "
                f"messages_30d={group.get('messages_30d')}"
            )
        if member:
            lines.append(
                f"- user's role={member.get('role')}, muted_by_user={member.get('group_muted_by_user')}, "
                f"reads_30d={member.get('messages_read_30d')}, replies_30d={member.get('replies_sent_30d')}, "
                f"dismissed_30d={member.get('notifications_dismissed_30d')}"
            )
            if member.get("group_muted_by_user") == "1":
                lines.append(
                    "  -> User has MUTED this group. Only an urgent/direct/important message should "
                    "override a mute; routine chatter should stay muted or digest."
                )
        if sender_user_id:
            sender_member = ds.membership(group_id, sender_user_id)
            if sender_member:
                lines.append(f"- sender's role in group: {sender_member.get('role')}")
                if sender_member.get("role") == "admin":
                    lines.append("  -> Sender is a group ADMIN: admin operational messages are usually higher-trust/priority.")

    elif conversation_type == "business" and business_id:
        biz = ds.business(business_id)
        hist = ds.biz_history(user_id, business_id)
        lines.append("\n## Business sender context")
        if biz:
            lines.append(
                f"- business='{biz.get('display_name')}' brand='{biz.get('brand_name')}', "
                f"category={biz.get('category')}, verified={biz.get('verified')}, "
                f"official_domain={biz.get('official_domain')}, domain_used_by_sender={biz.get('domain_used_by_sender')}, "
                f"account_age_days={biz.get('account_age_days')}, "
                f"domain_used_by_sender_age_days={biz.get('domain_used_by_sender_age_days')}, "
                f"user_reports_30d={biz.get('user_reports_30d')}"
            )
            if biz.get("official_domain") and biz.get("domain_used_by_sender") and \
               biz.get("official_domain") != biz.get("domain_used_by_sender"):
                lines.append(
                    f"  -> MISMATCH: sender is using domain '{biz.get('domain_used_by_sender')}' "
                    f"which does NOT match the business's official domain '{biz.get('official_domain')}'. "
                    "This is a strong scam/spoofing signal."
                )
            if biz.get("verified") != "1":
                lines.append("  -> Business account is NOT verified.")
            try:
                if int(biz.get("user_reports_30d") or 0) >= 5:
                    lines.append("  -> This business has a notably high number of user reports recently.")
            except ValueError:
                pass
        if hist:
            lines.append(
                f"- user relationship: why_known={hist.get('why_user_knows_account')}, "
                f"last_activity={hist.get('last_activity_at')}, allows_promotions={hist.get('allows_promotions')}, "
                f"promotions_opted_out_at={hist.get('promotions_opted_out_at') or 'n/a'}, "
                f"activity_180d={hist.get('activity_count_180d')}, opened_30d={hist.get('messages_opened_30d')}, "
                f"dismissed_30d={hist.get('messages_dismissed_30d')}, replied_30d={hist.get('messages_replied_30d')}"
            )
            if hist.get("promotions_opted_out_at"):
                lines.append("  -> User explicitly OPTED OUT of promotions from this business.")
            if hist.get("allows_promotions") == "0":
                lines.append("  -> User does not allow promotional messages from this business.")
        else:
            lines.append("- No prior relationship on record between this user and this business.")

    elif conversation_type == "personal":
        lines.append("\n## Personal (1:1) context")
        lines.append(f"- direct message from sender_user_id={sender_user_id}")

    # --- forwarding signal ---
    if forwarded_count and forwarded_count > 0:
        lines.append(f"\n## Forwarding\n- forwarded_count={forwarded_count} (message has been forwarded; chain forwards often correlate with low-value broadcast content or misinformation).")

    # --- retrieved historical evidence ---
    evidence = retriever.top_evidence(
        user_id=user_id,
        message_text=message_text,
        sender_user_id=sender_user_id,
        group_id=group_id,
        business_id=business_id,
        media_type=media_type,
    )
    lines.append("\n## Relevant history for this user (most similar/related past messages)")
    if evidence:
        for ev in evidence:
            snippet = (ev.row.get("message_text") or "")[:160].replace("\n", " ")
            if not snippet:
                snippet = f"[{ev.row.get('media_type','media')} message, no text]"
            lines.append(
                f"- {ev.message_id} (score={ev.score:.2f}, {ev.row.get('created_at')}): "
                f"\"{snippet}\" -> user reaction: {_fmt_event(ev.event)}"
            )
    else:
        lines.append("- No sufficiently similar or related historical message found for this user.")

    # --- media ---
    media_path = None
    if media_type in ("image", "voice") and media_id:
        media_path = ds.media_path(media_type, media_id)
        lines.append(f"\n## Media\n- This message includes a {media_type} attachment. It is provided separately for you to inspect directly.")

    prompt_context_block = "\n".join(lines)

    return MessageContext(
        message_id=msg["message_id"],
        conversation_type=conversation_type,
        message_text=message_text,
        media_type=media_type,
        media_path=media_path,
        forwarded_count=forwarded_count,
        prompt_context_block=prompt_context_block,
        evidence=evidence,
    )

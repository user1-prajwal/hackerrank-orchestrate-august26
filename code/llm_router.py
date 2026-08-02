"""
llm_router.py

Wraps the Gemini API call: builds the multimodal request (text + optional
image/audio file), calls the model, and validates/repairs the JSON
response against our schema. Retries with backoff on rate limits or
transient errors, and falls back to a safe default if the model call
keeps failing (so the pipeline never crashes mid-run and always produces
a row for every message).
"""

from __future__ import annotations

import json
import mimetypes
import os
import re
import time

from google import genai
from google.genai import types

from prompts import SYSTEM_INSTRUCTIONS, build_user_prompt

VALID_ACTIONS = {"notify", "digest", "mute"}
VALID_TYPES = {
    "personal", "urgent", "event", "payment", "business_update",
    "promotion", "greeting", "forward", "spam", "scam", "unknown",
}

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


class RoutingResult:
    def __init__(self, action, message_type, reason, confidence, evidence_message_ids):
        self.action = action
        self.message_type = message_type
        self.reason = reason
        self.confidence = confidence
        self.evidence_message_ids = evidence_message_ids

    def to_row(self, message_id: str) -> dict:
        ev = self.evidence_message_ids
        ev_str = ";".join(ev) if ev else "none"
        conf = self.confidence
        try:
            conf = round(float(conf), 2)
        except (TypeError, ValueError):
            conf = 0.5
        return {
            "message_id": message_id,
            "action": self.action,
            "message_type": self.message_type,
            "reason": self.reason,
            "confidence": conf,
            "evidence_message_ids": ev_str,
        }


def _fallback_result(reason: str) -> RoutingResult:
    return RoutingResult(
        action="digest",
        message_type="unknown",
        reason=reason,
        confidence=0.3,
        evidence_message_ids=[],
    )


def _parse_model_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text.strip())
    text = re.sub(r"```$", "", text.strip())
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = _JSON_BLOCK_RE.search(text)
        if m:
            return json.loads(m.group(0))
        raise


def _validate(parsed: dict, known_history_ids: set[str]) -> RoutingResult:
    action = str(parsed.get("action", "")).strip().lower()
    if action not in VALID_ACTIONS:
        action = "digest"

    message_type = str(parsed.get("message_type", "")).strip().lower()
    if message_type not in VALID_TYPES:
        message_type = "unknown"

    reason = str(parsed.get("reason", "")).strip() or "No reason provided by model."
    # Keep reasons short and single-sentence-ish.
    if len(reason) > 300:
        reason = reason[:297].rstrip() + "..."

    try:
        confidence = float(parsed.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))
    except (TypeError, ValueError):
        confidence = 0.5

    ev = parsed.get("evidence_message_ids", [])
    if isinstance(ev, str):
        ev = [e.strip() for e in ev.split(";") if e.strip() and e.strip().lower() != "none"]
    if not isinstance(ev, list):
        ev = []
    # Only keep IDs that actually exist in message_history, to avoid the
    # model hallucinating evidence IDs that don't exist.
    ev = [e for e in ev if e in known_history_ids]

    return RoutingResult(action, message_type, reason, confidence, ev)


class GeminiRouter:
    def __init__(self, api_key: str | None = None, model: str = "gemini-3.5-flash-lite", max_retries: int = 4):
        api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY environment variable is not set. "
                "Get a free key at https://aistudio.google.com/apikey and export it, e.g.\n"
                "  export GEMINI_API_KEY=your_key_here"
            )
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.max_retries = max_retries

    def _load_media_part(self, media_path: str) -> types.Part | None:
        if not media_path or not os.path.exists(media_path):
            return None
        mime, _ = mimetypes.guess_type(media_path)
        if not mime:
            mime = "image/jpeg" if media_path.lower().endswith((".jpg", ".jpeg", ".png")) else "audio/mpeg"
        with open(media_path, "rb") as f:
            data = f.read()
        return types.Part.from_bytes(data=data, mime_type=mime)

    def route(
        self,
        *,
        context_block: str,
        message_text: str,
        media_type: str,
        media_path: str | None,
        known_history_ids: set[str],
    ) -> RoutingResult:
        user_prompt = build_user_prompt(context_block, message_text, media_type)

        parts: list = [types.Part.from_text(text=user_prompt)]
        media_part = self._load_media_part(media_path) if media_path else None
        if media_part is not None:
            parts.append(media_part)

        contents = [types.Content(role="user", parts=parts)]

        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTIONS,
            temperature=0.2,
            response_mime_type="application/json",
        )

        last_err = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=config,
                )
                raw_text = response.text or ""
                parsed = _parse_model_json(raw_text)
                return _validate(parsed, known_history_ids)
            except Exception as e:  # noqa: BLE001 - want to catch/retry any transient API error
                last_err = e
                msg = str(e).lower()
                is_rate_limit = "429" in msg or "resource_exhausted" in msg or "quota" in msg
                is_transient = "500" in msg or "503" in msg or "timeout" in msg or is_rate_limit
                if attempt < self.max_retries and is_transient:
                    backoff = min(60, (2 ** attempt) + (attempt * 0.5))
                    print(f"  [retry {attempt}/{self.max_retries}] {type(e).__name__}: {e}. Waiting {backoff:.0f}s...")
                    time.sleep(backoff)
                    continue
                break

        return _fallback_result(f"LLM call failed after {self.max_retries} attempts ({last_err}); defaulted to safe digest.")

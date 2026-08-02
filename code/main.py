#!/usr/bin/env python3
"""
main.py

Entry point for the WhatsApp Message Notification Router.

Usage:
    export GEMINI_API_KEY=your_key_here
    python3 main.py --dataset ../dataset --output ../dataset/output.csv

Flags:
    --dataset      Path to the dataset directory (default: ../dataset)
    --output       Path to write the output CSV (default: <dataset>/output.csv)
    --model        Gemini model name (default: gemini-3.5-flash-lite)
    --limit N      Only process the first N messages (useful for smoke-testing)
    --dry-run      Skip Gemini calls entirely; use only the deterministic
                   safety rules + a heuristic fallback. Useful for testing
                   the full pipeline (context building, retrieval, CSV
                   writing) without spending any API quota, or when no
                   API key is available yet.
    --start-from   Resume from this message_id (skips everything before it
                   in the input order). Combined with --resume, lets you
                   continue a run that was interrupted partway through.
    --resume       If set, load any existing rows already in the output
                   file and skip message_ids that already have a
                   non-empty action, appending new results instead of
                   overwriting them.

The script is intentionally resumable and rate-limit-tolerant: since the
free Gemini tier can throttle mid-run, you can stop and re-run with
--resume and it will only process what's left.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import Dataset
from retrieval import Retriever
from context_builder import build_context
from safety_rules import evaluate_pre_llm
from rule_engine import evaluate_rules
from confidence import compute_confidence
from llm_router import GeminiRouter, RoutingResult

OUTPUT_COLUMNS = ["message_id", "action", "message_type", "reason", "confidence", "evidence_message_ids"]


def heuristic_fallback(msg: dict, ctx) -> RoutingResult:
    """
    Used only in --dry-run mode (no LLM), or as an extra safety net.
    A simple, explainable rule-of-thumb so the pipeline always produces
    a complete, valid output.csv even without API access -- this is NOT
    the primary decision system, just a way to validate the plumbing.
    """
    text = (msg.get("message_text") or "").lower()
    conv_type = msg["conversation_type"]

    if any(k in text for k in ("otp", "verify now", "blocked", "reattempt fee", "confirm password")):
        return RoutingResult("mute", "scam", "Heuristic fallback: message contains classic phishing/urgency language.", 0.55, [])

    if any(k in text for k in ("good morning", "forwarded as received", "fwd as received", "share with everyone")):
        return RoutingResult("mute", "forward", "Heuristic fallback: generic forwarded greeting chain message.", 0.5, [])

    if conv_type == "business":
        return RoutingResult("digest", "business_update", "Heuristic fallback: business message with no strong urgency signal.", 0.4, [])

    if conv_type == "group":
        return RoutingResult("digest", "unknown", "Heuristic fallback: group message with no strong urgency signal.", 0.4, [])

    return RoutingResult("notify", "personal", "Heuristic fallback: direct personal message, defaulting to notify.", 0.4, [])


def load_existing_output(path: str) -> dict[str, dict]:
    existing = {}
    if os.path.exists(path):
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("action"):  # already has a prediction
                    existing[row["message_id"]] = row
    return existing


def main():
    ap = argparse.ArgumentParser(description="WhatsApp Message Notification Router")
    ap.add_argument("--dataset", default="../dataset", help="Path to dataset directory")
    ap.add_argument("--output", default=None, help="Path to output CSV (default: <dataset>/output.csv)")
    ap.add_argument("--model", default="gemini-3.5-flash-lite", help="Gemini model name")
    ap.add_argument("--limit", type=int, default=None, help="Only process first N messages")
    ap.add_argument("--dry-run", action="store_true", help="Skip LLM calls; use heuristic fallback only")
    ap.add_argument("--resume", action="store_true", help="Resume from existing output.csv, skipping completed rows")
    ap.add_argument("--sleep", type=float, default=1.0, help="Seconds to sleep between LLM calls (free-tier RPM safety)")
    args = ap.parse_args()

    dataset_dir = os.path.abspath(args.dataset)
    output_path = args.output or os.path.join(dataset_dir, "output.csv")

    print(f"Loading dataset from {dataset_dir} ...")
    ds = Dataset.load(dataset_dir)
    print(f"  {len(ds.messages)} incoming messages, {len(ds.message_history)} historical messages, "
          f"{len(ds.users)} users, {len(ds.groups)} groups, {len(ds.businesses)} businesses.")

    retriever = Retriever(ds)
    known_history_ids = set(ds.message_history.keys())

    router = None
    if not args.dry_run:
        router = GeminiRouter(model=args.model)
        print(f"Using Gemini model: {args.model}")
    else:
        print("DRY RUN: no LLM calls will be made; using heuristic fallback + safety rules only.")

    existing = load_existing_output(output_path) if args.resume else {}
    if existing:
        print(f"Resuming: {len(existing)} messages already have predictions, will skip those.")

    messages = ds.messages
    if args.limit:
        messages = messages[: args.limit]

    results: dict[str, dict] = dict(existing)

    total = len(messages)
    stats = {"safety_override": 0, "rule_resolved": 0, "llm_used": 0}

    for i, msg in enumerate(messages, 1):
        mid = msg["message_id"]
        if mid in results:
            continue

        print(f"[{i}/{total}] {mid} ...", end=" ", flush=True)

        # 1. Deterministic safety pre-check. Overrides everything else on
        #    clear, unambiguous scam patterns (see safety_rules.py).
        verdict = evaluate_pre_llm(ds, msg)

        if verdict.forced:
            stats["safety_override"] += 1
            row = RoutingResult(
                verdict.action, verdict.message_type, verdict.reason,
                verdict.confidence, []
            ).to_row(mid)
            print(f"-> {row['action']}/{row['message_type']} (safety override)")
            results[mid] = row
            _write_output(output_path, ds.messages, results)
            continue

        # 2. Retrieval always runs -- both the rule engine and the LLM
        #    context need it, and confidence scoring needs it too.
        ctx = build_context(ds, retriever, msg)

        # 3. Lightweight rule engine: resolves routine/clear-cut cases
        #    without spending an LLM call. Always defers on image/voice
        #    messages, since reading the attachment needs the model.
        rule_verdict = evaluate_rules(ds, msg, ctx.evidence)

        if rule_verdict.resolved:
            stats["rule_resolved"] += 1
            conf = compute_confidence(ds, msg, ctx.evidence, llm_confidence=None)
            evidence_ids = [e.message_id for e in ctx.evidence[:2]] if ctx.evidence else []
            row = RoutingResult(
                rule_verdict.action, rule_verdict.message_type, rule_verdict.reason,
                conf.value, evidence_ids
            ).to_row(mid)
            print(f"-> {row['action']}/{row['message_type']} (rule engine, conf={row['confidence']})")
            results[mid] = row
            _write_output(output_path, ds.messages, results)
            continue

        # 4. Fell through to Gemini: either genuinely ambiguous, or a
        #    media (image/voice) message that needs multimodal reading.
        if args.dry_run:
            result = heuristic_fallback(msg, ctx)
        else:
            stats["llm_used"] += 1
            result = router.route(
                context_block=ctx.prompt_context_block,
                message_text=ctx.message_text,
                media_type=ctx.media_type,
                media_path=ctx.media_path,
                known_history_ids=known_history_ids,
            )
            time.sleep(args.sleep)

        # Confidence is computed deterministically first; the LLM's own
        # confidence is only used as a fallback signal inside compute_confidence
        # when deterministic evidence is weak (see confidence.py).
        conf = compute_confidence(ds, msg, ctx.evidence, llm_confidence=result.confidence)
        result.confidence = conf.value

        row = result.to_row(mid)
        basis_note = "llm+det.conf" if conf.deterministic else "llm+llm.conf"
        print(f"-> {row['action']}/{row['message_type']} (conf={row['confidence']}, {basis_note})")

        results[mid] = row
        _write_output(output_path, ds.messages, results)

    print(f"\nDone. Wrote {len(results)} rows to {output_path}")
    print(f"  Safety overrides: {stats['safety_override']}")
    print(f"  Rule-engine resolved (no LLM call): {stats['rule_resolved']}")
    print(f"  Sent to Gemini: {stats['llm_used']}")


def _write_output(output_path: str, all_messages: list[dict], results: dict[str, dict]):
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for msg in all_messages:
            mid = msg["message_id"]
            if mid in results:
                writer.writerow(results[mid])


if __name__ == "__main__":
    main()

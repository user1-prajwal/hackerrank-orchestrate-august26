#!/usr/bin/env python3
"""
evaluate_samples.py

Runs the router against dataset/sample_messages.csv (which has known
correct action/message_type/evidence labels) and reports accuracy.
This lets you sanity-check quality BEFORE spending API quota on the
full messages.csv run.

Usage:
    export GEMINI_API_KEY=your_key_here
    python3 evaluate_samples.py --dataset ../dataset
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="../dataset")
    ap.add_argument("--model", default="gemini-3.5-flash-lite")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sleep", type=float, default=1.0)
    args = ap.parse_args()

    dataset_dir = os.path.abspath(args.dataset)
    ds = Dataset.load(dataset_dir)
    retriever = Retriever(ds)
    known_history_ids = set(ds.message_history.keys())
    router = GeminiRouter(model=args.model)

    sample_path = os.path.join(dataset_dir, "sample_messages.csv")
    with open(sample_path, newline="", encoding="utf-8") as f:
        samples = list(csv.DictReader(f))

    if args.limit:
        samples = samples[: args.limit]

    n = len(samples)
    action_correct = 0
    type_correct = 0
    both_correct = 0
    llm_used = 0
    rule_resolved = 0
    safety_override = 0

    for i, s in enumerate(samples, 1):
        msg = {
            "message_id": s["message_id"],
            "user_id": s["user_id"],
            "conversation_type": s["conversation_type"],
            "group_id": s.get("group_id") or "",
            "business_id": s.get("business_id") or "",
            "sender_user_id": s.get("sender_user_id") or "",
            "created_at": s.get("created_at") or "",
            "message_text": s.get("message_text") or "",
            "media_type": s.get("media_type") or "",
            "media_id": s.get("media_id") or "",
            "forwarded_count": s.get("forwarded_count") or "0",
        }

        verdict = evaluate_pre_llm(ds, msg)
        if verdict.forced:
            safety_override += 1
            pred_action, pred_type = verdict.action, verdict.message_type
        else:
            ctx = build_context(ds, retriever, msg)
            rule_verdict = evaluate_rules(ds, msg, ctx.evidence)

            if rule_verdict.resolved:
                rule_resolved += 1
                pred_action, pred_type = rule_verdict.action, rule_verdict.message_type
            else:
                llm_used += 1
                result = router.route(
                    context_block=ctx.prompt_context_block,
                    message_text=ctx.message_text,
                    media_type=ctx.media_type,
                    media_path=ctx.media_path,
                    known_history_ids=known_history_ids,
                )
                pred_action, pred_type = result.action, result.message_type
                time.sleep(args.sleep)

        true_action, true_type = s["action"], s["message_type"]
        a_ok = pred_action == true_action
        t_ok = pred_type == true_type
        action_correct += a_ok
        type_correct += t_ok
        both_correct += a_ok and t_ok

        flag = "OK " if a_ok and t_ok else ("action" if a_ok else ("type" if t_ok else "MISS"))
        print(f"[{i}/{n}] {s['message_id']}: pred=({pred_action},{pred_type}) "
              f"true=({true_action},{true_type}) [{flag}]")

    print(f"\nAction accuracy:  {action_correct}/{n} = {action_correct/n:.1%}")
    print(f"Type accuracy:    {type_correct}/{n} = {type_correct/n:.1%}")
    print(f"Both correct:     {both_correct}/{n} = {both_correct/n:.1%}")
    print(f"\nSafety overrides: {safety_override}/{n}")
    print(f"Rule-engine resolved (no LLM call): {rule_resolved}/{n}")
    print(f"Sent to Gemini: {llm_used}/{n}")


if __name__ == "__main__":
    main()

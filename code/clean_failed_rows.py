#!/usr/bin/env python3
"""
clean_failed_rows.py

Removes rows from output.csv whose `reason` field is actually a dumped
API error/traceback rather than a real classification (i.e. rows where
llm_router.py exhausted its retries and fell back to the safe-default
digest). This happens when the free-tier daily/per-minute quota is hit
mid-run.

Run this BEFORE `python3 main.py --resume` after a quota-interrupted run.
Plain --resume can't tell a real "digest" decision apart from a failed-
retry fallback that also happens to say "digest" -- both have a non-empty
`action` field, so --resume alone would wrongly treat the failed rows as
already done and skip them forever. This script removes only the rows
that are genuinely bad, so --resume then correctly retries just those.

Usage:
    python3 clean_failed_rows.py --output ../dataset/output.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil

FAILURE_MARKERS = (
    "LLM call failed after",
    "RESOURCE_EXHAUSTED",
    "ClientError",
    "quotaMetric",
)


def is_failed_row(row: dict) -> bool:
    reason = row.get("reason", "") or ""
    return any(marker in reason for marker in FAILURE_MARKERS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="../dataset/output.csv", help="Path to output.csv to clean")
    ap.add_argument("--no-backup", action="store_true", help="Skip writing a .bak backup copy")
    args = ap.parse_args()

    if not os.path.exists(args.output):
        print(f"No file found at {args.output}, nothing to clean.")
        return

    with open(args.output, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    good_rows = [r for r in rows if not is_failed_row(r)]
    bad_rows = [r for r in rows if is_failed_row(r)]

    if not bad_rows:
        print(f"No failed-fallback rows found in {args.output}. Nothing to clean; safe to run --resume directly.")
        return

    if not args.no_backup:
        backup_path = args.output + ".bak"
        shutil.copy(args.output, backup_path)
        print(f"Backed up original to {backup_path}")

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in good_rows:
            writer.writerow(r)

    print(f"Removed {len(bad_rows)} failed-fallback row(s):")
    for r in bad_rows:
        print(f"  - {r['message_id']}")
    print(f"\nKept {len(good_rows)} good row(s) in {args.output}.")
    print("Now run: python3 main.py --dataset ../dataset --output ../dataset/output.csv --resume --model gemini-3.5-flash-lite")


if __name__ == "__main__":
    main()

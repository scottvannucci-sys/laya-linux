#!/usr/bin/env python3
"""Build a laya-linux routing eval set from a Hermes kanban board.

Reads done tasks (title + body) from the board's SQLite DB, calls the local
laya-linux server's router preset on each, and compares the model's domain
choice against the assignee each card actually went to (the human/system
routing decision). Writes JSONL to benchmarks/eval/.

Ground truth caveat: the laya `router` preset's domain labels (code,
writing, data_analysis, ...) don't map 1:1 to forge-* profile names, so the
comparison is profile-class -> expected domain (see PROFILE_DOMAIN). Cards
whose profile has no mapping are scored as 'unmapped' and reported
separately.

Usage:
    python3 scripts/build_routing_eval.py \
        --board /home/scott/.hermes/kanban/boards/universal-assistant/kanban.db \
        --limit 300 --out benchmarks/eval/routing_eval.jsonl
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
import urllib.request
from pathlib import Path

# forge-* profile class -> laya router domain label.
# ponytail: crude heuristic map, refine split by title keywords if the
# confusion matrix shows a big class bleeding into another.
PROFILE_DOMAIN = {
    "forge-backend": "code",
    "forge-frontend": "code",
    "forge-architect": "code",
    "forge-architect-codex": "code",
    "forge-integration": "code",
    "forge-tooling": "code",
    "forge-security": "code",
    "forge-verifier": "code",
    "forge-verifier-local": "code",
    "forge-data": "data_analysis",
    "forge-documentation": "writing",
    "forge-release": "code",
    "forge-orchestrator": None,  # planning/routing, no laya domain equivalent
    "default": None,
}


def fetch_tasks(db_path: str, limit: int) -> list[dict]:
    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    rows = db.execute(
        """SELECT id, title, body, assignee FROM tasks
           WHERE status='done' AND assignee IS NOT NULL
             AND length(body) > 40
           ORDER BY completed_at DESC LIMIT ?""",
        (limit,),
    ).fetchall()
    return [{"id": r[0], "title": r[1], "body": r[2], "assignee": r[3]} for r in rows]


def predict(server: str, token: str, text: str, timeout: float = 30.0) -> dict:
    body = json.dumps({
        "model": "typed",
        "state": {"request": text},
        "questions": ROUTER_QUESTIONS,
    }).encode()
    req = urllib.request.Request(
        f"{server}/v1/predict", data=body, method="POST",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


# Mirror of laya router preset (keep in sync with src/laya_linux/presets.py).
ROUTER_QUESTIONS = {
    "domain": {
        "type": "choice",
        "instructions": "Which domain does `request` belong to?",
        "criteria": {
            "code": "software programming, debugging, or codebase tasks",
            "math_or_logic": "math, formal logic or algorithm puzzles",
            "writing": "prose, documentation, marketing or communication",
            "factual_lookup": "simple factual questions or lookups",
            "data_analysis": "analyzing datasets, statistics or trends",
            "chitchat": "casual conversation with no task",
        },
    },
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", required=True)
    ap.add_argument("--server", default="http://192.168.50.207:8142")
    ap.add_argument("--token-file", default="/home/scott/.laya/token")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--out", default="benchmarks/eval/routing_eval.jsonl")
    ap.add_argument("--sleep", type=float, default=0.0)
    ap.add_argument("--client-timeout", type=float, default=30.0)
    ap.add_argument("--retry-errors", action="store_true",
                    help="Re-predict only ERROR rows already in --out")
    args = ap.parse_args()

    token = Path(args.token_file).read_text().strip()
    if "=" in token and "\n" not in token:
        token = token.split("=", 1)[1].strip().strip('"').strip("'")

    tasks = fetch_tasks(args.board, args.limit)
    by_id = {t["id"]: t for t in tasks}

    if args.retry_errors:
        prev = [json.loads(ln) for ln in Path(args.out).read_text().splitlines() if ln.strip()]
        bad_ids = [r["task_id"] for r in prev if r["predicted_domain"].startswith("ERROR")]
        print(f"retrying {len(bad_ids)} error rows", file=sys.stderr)
        targets = [by_id[i] for i in bad_ids if i in by_id]
        # Keep the good rows; retry results replace the error rows below.
        prev_good = [r for r in prev if not r["predicted_domain"].startswith("ERROR")]
    else:
        targets = tasks
        prev_good = []
    print(f"loaded {len(targets)} done tasks", file=sys.stderr)

    out_rows = []
    for i, t in enumerate(targets):
        expected = PROFILE_DOMAIN.get(t["assignee"], "unmapped")
        if expected is None:
            continue  # orchestrator/default cards: no laya domain equivalent
        text = f"{t['title']}\n\n{t['body'][:2000]}"
        try:
            pred = predict(args.server, token, text, args.client_timeout)
            got = pred["answers"]["domain"]["choice"]
            conf = pred["answers"]["domain"]["confidence"]
        except Exception as e:
            got, conf = f"ERROR:{type(e).__name__}", None
        out_rows.append({
            "task_id": t["id"], "assignee": t["assignee"],
            "expected_domain": expected, "predicted_domain": got,
            "confidence": conf,
        })
        if args.sleep:
            time.sleep(args.sleep)
        if (i + 1) % 25 == 0:
            print(f"{i+1}/{len(tasks)}", file=sys.stderr)

    out_rows = prev_good + out_rows

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for row in out_rows:
            f.write(json.dumps(row) + "\n")

    scored = [r for r in out_rows if not r["predicted_domain"].startswith("ERROR")]
    errors = len(out_rows) - len(scored)
    if scored:
        correct = sum(1 for r in scored if r["predicted_domain"] == r["expected_domain"])
        print(f"scored {len(scored)} ({errors} request errors)")
        print(f"agreement: {correct}/{len(scored)} = {correct/len(scored):.1%}")
        for dom in sorted({r["expected_domain"] for r in scored}):
            sub = [r for r in scored if r["expected_domain"] == dom]
            c = sum(1 for r in sub if r["predicted_domain"] == dom)
            print(f"  {dom}: {c}/{len(sub)} = {c/len(sub):.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Consolidate the Campaign-3 workflow result into a compact report.

Usage: python consolidate.py <workflow_output.json> [--full]
The output file is the harness task-output JSON with a "result" field
(the workflow's return value: {verify, survey, review, stats}).
"""
import json
import sys
from collections import Counter

SEV_ORDER = {"BUG": 0, "DISCREPANCY": 1, "WEAK_TEST": 2, "CITATION": 3, "DOC_GAP": 4,
             "CRITICAL": 0, "MAJOR": 1, "MINOR": 2, "NIT": 3}


def load(path):
    doc = json.load(open(path))
    res = doc.get("result", doc)
    if isinstance(res, str):
        res = json.loads(res)
    return doc, res


def main():
    path = sys.argv[1]
    full = "--full" in sys.argv
    doc, res = load(path)
    verify = [v for v in (res.get("verify") or []) if v]
    survey = res.get("survey")
    review = [r for r in (res.get("review") or []) if r]

    print(f"# Campaign 3 consolidated ({len(verify)} task verdicts)")
    print(f"workflow agents: {doc.get('agentCount')} | tokens: {doc.get('totalTokens')}")

    # --- verify table ---
    n_conf = sum(1 for v in verify if v.get("verdict") == "CONFIRMED_CORRECT")
    survived, refuted_all = [], []
    for v in verify:
        t = v.get("task", "?").split(" ")[0]
        for j in v.get("judged", []) or []:
            votes = j.get("votes") or []
            rec = dict(j, task=t, nvotes=len(votes))
            if j.get("survives") and votes:
                survived.append(rec)
            elif votes:
                refuted_all.append(rec)
            else:
                # no votes recorded (refuters missing) -> unresolved
                rec["unresolved"] = True
                survived.append(rec)
    print(f"\nCONFIRMED_CORRECT: {n_conf}/{len(verify)}")
    print(f"surviving findings (>=1 refuter agrees or unresolved): {len(survived)}")
    print(f"refuted findings: {len(refuted_all)}")
    print("severity mix (surviving):", dict(Counter(f['severity'] for f in survived)))

    print("\n## Per-task verdicts")
    for v in sorted(verify, key=lambda x: x.get("task", "")):
        t = v.get("task", "?").split(" ")[0]
        surv = [j for j in (v.get("judged") or [])
                if j.get("survives") or not (j.get("votes") or [])]
        sevs = ",".join(sorted({s['severity'] for s in surv}, key=lambda s: SEV_ORDER.get(s, 9))) or "-"
        pt = (v.get("parity_tests") or {}).get("status", "?")
        print(f"{t:46s} {v.get('verdict','?'):10s} tests={pt:5s} "
              f"strength={v.get('test_strength','?'):10s} cit={str(v.get('citations_ok'))[0]} "
              f"surviving=[{sevs}]")

    print("\n## Surviving findings (most severe first)")
    for f in sorted(survived, key=lambda f: (SEV_ORDER.get(f["severity"], 9), f["task"])):
        tag = "UNRESOLVED(no refuter votes)" if f.get("unresolved") else \
              ("unanimous" if f.get("unanimous") else f"{sum(1 for v0 in f['votes'] if not v0.get('refuted'))}/{f['nvotes']} not-refuted")
        print(f"\n[{f['severity']}] {f['task']} — {f['title']}  ({tag})")
        print(f"  {f.get('file','?')}:{f.get('line','?')}")
        print(f"  evidence: {f.get('evidence','')[:400]}")
        if f.get("suggested_fix"):
            print(f"  fix: {f['suggested_fix'][:300]}")
        if full:
            for v0 in f.get("votes") or []:
                print(f"    vote(refuted={v0.get('refuted')}): {v0.get('reasoning','')[:300]}")

    if full and refuted_all:
        print("\n## Refuted findings (for the record)")
        for f in sorted(refuted_all, key=lambda f: (SEV_ORDER.get(f["severity"], 9), f["task"])):
            print(f"[{f['severity']}] {f['task']} — {f['title']}")
            for v0 in f.get("votes") or []:
                print(f"    refuted={v0.get('refuted')}: {v0.get('reasoning','')[:250]}")

    # --- survey ---
    print("\n\n# Gap survey")
    if survey and survey.get("synthesis"):
        s = survey["synthesis"]
        print("## MUST-ADD")
        for c in s.get("must_add", []):
            print(f"- {c['name']} (arXiv:{c['arxiv']}) {c['repo']} verified={c.get('verified')}")
            print(f"    {c.get('rationale','')[:350]}")
        print("## NICE-TO-HAVE")
        for c in s.get("nice_to_have", []):
            print(f"- {c['name']} (arXiv:{c['arxiv']}) {c['repo']}")
            print(f"    {c.get('rationale','')[:250]}")
        print("## rejected/dupes:", "; ".join(s.get("rejected_duplicates", []))[:600])
        print("## notes:", s.get("notes", "")[:800])
        print(f"(raw candidates from lenses: {len((survey.get('raw') or []))})")
    else:
        print("(no synthesis present)")

    # --- pipeline review ---
    print("\n\n# Pipeline review")
    for r in review:
        print(f"\n## {r.get('dimension','?')} — {r.get('summary','')[:300]}")
        for j in r.get("judged", []) or []:
            ok = "SURVIVES" if j.get("survives") else "refuted"
            print(f"  [{j['severity']}][{ok}] {j['title']} ({j.get('file','?')}:{j.get('line','?')})")
            print(f"      {j.get('evidence','')[:300]}")
            if j.get("survives") and j.get("suggested_fix"):
                print(f"      fix: {j['suggested_fix'][:250]}")
            if full and j.get("vote"):
                print(f"      vote: refuted={j['vote'].get('refuted')} — {j['vote'].get('reasoning','')[:250]}")
        for m in r.get("minors", []) or []:
            print(f"  [minor:{m['severity']}] {m['title']} ({m.get('file','?')})")


if __name__ == "__main__":
    main()

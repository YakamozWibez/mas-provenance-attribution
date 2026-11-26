"""Pre-registration 04: k* ("filtered" ambiguity), computed from the stripped trace alone
(no model calls, no judge outputs).

Start from the k look-alike candidates of ambiguity.py (stripped view: same numbers and
arithmetic symbols, and the same declared sources). Then remove candidates that a careful
reader can clear, WITH RESPECT TO THE ERRORS WE INJECT (one value changed: fabricated U or
swapped M). Clearing is not proof that an element is correct in general: a step with
correct arithmetic can still be a reasoning error (wrong quantities), which we do not inject.

  Test A, arithmetic: a candidate whose text contains a checked computation
     "expression = value" that is arithmetically consistent is cleared (a changed value
     makes the computation inconsistent).
  Test B, order: the numbers of agent 1 are aligned, in order, with the numbers of the task
     (digits and number words). A candidate of agent 1 whose numbers all fall on aligned
     equal positions is cleared. Alignment: global, match +2, mismatch -1, gap -1, ties
     resolved towards the diagonal.

  k_starA = candidates left after test A
  k_starB = candidates left after tests A and B
  If a test clears the injected element itself, culprit_cleared_* is true (the cue misled).

  python ambiguity_star.py RESULTS.jsonl > AMBSTAR.jsonl
"""
import json
import re
import sys

from judge import strip_text
from r2.families import WORDS, TENS, arithmetic, norm_num

MULT = {"twice": "2", "double": "2", "doubled": "2", "triple": "3", "tripled": "3",
        "half": "0.5", "dozen": "12"}
TOKEN = re.compile(r"\$?\d[\d,]*(?:\.\d+)?%?|[A-Za-z]+(?:-[A-Za-z]+)?")


def ordered_numbers(text, words=True):
    out = []
    for tok in TOKEN.findall(text or ""):
        if tok[0].isdigit() or tok[0] == "$":
            v = norm_num(tok)
            if v is not None:
                out.append(v)
        elif words:
            low = tok.lower()
            if "-" in low:
                a, b = low.split("-", 1)
                tens = dict(TENS, twenty=20)
                if a in tens and b in WORDS and WORDS[b] < 10:
                    out.append(str(tens[a] + WORDS[b]))
                    continue
            if low in WORDS:
                out.append(str(WORDS[low]))
            elif low in MULT:
                out.append(MULT[low])
    return out


def align(a, b):
    """Global alignment of sequences a (task) and b (agent). Returns, for each index of b,
    True if it is aligned to an equal element of a."""
    n, m = len(a), len(b)
    S = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        S[i][0] = -i
    for j in range(1, m + 1):
        S[0][j] = -j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            S[i][j] = max(S[i - 1][j - 1] + (2 if a[i - 1] == b[j - 1] else -1),
                          S[i - 1][j] - 1, S[i][j - 1] - 1)
    ok = [False] * m
    i, j = n, m
    while i > 0 and j > 0:
        if S[i][j] == S[i - 1][j - 1] + (2 if a[i - 1] == b[j - 1] else -1):
            ok[j - 1] = a[i - 1] == b[j - 1]
            i, j = i - 1, j - 1
        elif S[i][j] == S[i - 1][j] - 1:
            i -= 1
        else:
            j -= 1
    return ok


def signature(e, with_sources):
    return (strip_text(e["text"]), tuple(sorted(e.get("sources") or ())) if with_sources else ())


def candidates(messages):
    for m in (1, 2):
        if m < len(messages):
            for e in messages[m]["elements"]:
                if e.get("final") or e["id"].endswith(".R"):
                    continue
                yield m, e


def passes_A(e):
    consistent, computed = arithmetic(strip_text(e["text"]))
    return consistent and bool(computed) and "=" in e["text"]


def k_star(run, element_id, with_sources):
    msgs = run["messages"]
    els = list(candidates(msgs))
    target = next((e for _, e in els if e["id"] == element_id), None)
    if target is None:
        return None
    sig = signature(target, with_sources)
    cands = [(m, e) for m, e in els if signature(e, with_sources) == sig]
    k = len(cands)
    # test A
    left_A = [(m, e) for m, e in cands if not passes_A(e)]
    # test B: align agent-1 numbers with task numbers
    task_nums = []
    for e in msgs[0]["elements"]:
        task_nums += ordered_numbers(e["text"], words=True)
    seq, owner = [], []
    for m, e in els:
        if m == 1:
            for v in ordered_numbers(strip_text(e["text"]), words=False):
                seq.append(v)
                owner.append(e["id"])
    ok = align(task_nums, seq)
    cleared_B = set()
    for eid in {o for o in owner}:
        flags = [ok[i] for i, o in enumerate(owner) if o == eid]
        if flags and all(flags):
            cleared_B.add(eid)
    left_B = [(m, e) for m, e in left_A if not (m == 1 and e["id"] in cleared_B)]
    ids_A, ids_B = {e["id"] for _, e in left_A}, {e["id"] for _, e in left_B}
    return {"k": k, "k_starA": len(left_A), "k_starB": len(left_B),
            "culprit_cleared_A": element_id not in ids_A,
            "culprit_cleared_B": element_id not in ids_B}


def main(path):
    for line in open(path):
        r = json.loads(line)
        if r.get("ok_major") is False:
            continue
        prov = r["cond"].endswith("+prov")
        for kind in ("U", "M"):
            I = (r.get("inj") or {}).get(kind) or {}
            if not (I.get("decisive") and I.get("run")):
                continue
            res = k_star(I["run"], I["element"], prov)
            if res is None:
                continue
            print(json.dumps({"task": r["task"], "cond": r["cond"], "kind": kind,
                              "element": I["element"], **res}))


if __name__ == "__main__":
    main(sys.argv[1])

"""Pre-registration 03: ambiguity k of each injected error, computed from the trace alone
(no model calls, no judge outputs).

k = number of elements in the agent messages that could have introduced the error (agent 1
and agent 2; reasoning paragraphs and final answers excluded) and that look IDENTICAL to the
injected element in the view the attributor gets. The injected element itself counts, so k >= 1.

What "identical" means in each view:
  full view:     the element's text (lowercased, whitespace normalised) + its declared sources
  stripped view: only numbers and arithmetic symbols (judge.strip_text) + its declared sources
Declared sources exist only in the +prov conditions.

An attributor that only has this signature cannot tell the k candidates apart, so its best
expected accuracy is 1/k. Order and position of elements are NOT part of the signature (see
prereg/02 and prereg/03 for what a violation of the bound would mean).

  python ambiguity.py results/math_gsm8k_Qwen-Qwen3.5-9B.jsonl  > results/..._ambiguity.jsonl
"""
import json
import re
import sys

from judge import strip_text


def signature(e, view, with_sources):
    text = e["text"]
    if view == "strip":
        text = strip_text(text)
    else:
        text = re.sub(r"\s+", " ", text.lower()).strip()
    src = tuple(sorted(e.get("sources") or ())) if with_sources else ()
    return (text, src)


def candidates(messages):
    for m in (1, 2):
        if m < len(messages):
            for e in messages[m]["elements"]:
                if e.get("final") or e["id"].endswith(".R"):
                    continue
                yield e


def k_of(run, element_id, view, with_sources):
    els = list(candidates(run["messages"]))
    target = next((e for e in els if e["id"] == element_id), None)
    if target is None:
        return None
    s = signature(target, view, with_sources)
    return sum(1 for e in els if signature(e, view, with_sources) == s)


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
            print(json.dumps({"task": r["task"], "cond": r["cond"], "kind": kind,
                              "element": I["element"],
                              "k_full": k_of(I["run"], I["element"], "full", prov),
                              "k_strip": k_of(I["run"], I["element"], "strip", prov)}))


if __name__ == "__main__":
    main(sys.argv[1])

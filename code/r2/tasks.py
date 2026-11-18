"""Task loaders.

- synth_math / synth_qa: synthetic tasks, ONLY to test the code with mocks.
- gsm8k: OpenAI grade-school-math test set (MIT licence, verified in the repo
  LICENSE). File: grade_school_math/data/test.jsonl from github.com/openai/grade-school-math
- hotpot: HotpotQA dev, distractor setting (CC BY-SA 4.0, verified in the
  official README). File: hotpot_dev_distractor_v1.json from the link in
  github.com/hotpotqa/hotpot (README, section "Data Download").
"""
import json
import random
import re

NAMES = ["Anna", "Luca", "Sara", "Marco", "Giulia", "Paolo"]
ITEMS = ["apples", "books", "stamps", "marbles", "cards"]


def synth_math(n, seed=0):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        name, item = rng.choice(NAMES), rng.choice(ITEMS)
        a, b, c = rng.sample(range(3, 60), 3)
        verbs = rng.sample([("buys", 1), ("finds", 1), ("gives away", -1), ("loses", -1)], 2)
        q, ans = f"{name} has {a} {item}.", a
        for (v, s), x in zip(verbs, (b, c)):
            q += f" Then {name} {v} {x} {item}."
            ans += s * x
        q += f" How many {item} does {name} have now?"
        out.append({"id": f"synth_math_{i:03d}", "question": q, "answer": ans})
    return out


SYL = ["Vel", "Mor", "Kas", "Trin", "Ald", "Oro", "Bri", "Sen", "Tal", "Qui", "Dor", "Mex", "Ula", "Ren", "Pax"]


def _name(rng, used):
    while True:
        n = rng.choice(SYL) + rng.choice(SYL).lower()
        if n not in used:
            used.add(n)
            return n


def synth_qa(n, seed=0):
    """Fictional two-hop questions with distractors."""
    rng = random.Random(seed)
    out = []
    for i in range(n):
        used = set()
        person, place, region = _name(rng, used), _name(rng, used), _name(rng, used)
        d_person, d_place, d_region = _name(rng, used), _name(rng, used), _name(rng, used)
        ctx = [(person, [f"{person} was born in {place}.", f"{person} wrote three novels."]),
               (place, [f"{place} is a town in the region of {region}.", f"{place} has a river port."]),
               (d_person, [f"{d_person} was born in {d_place}."]),
               (d_place, [f"{d_place} is a town in the region of {d_region}."])]
        rng.shuffle(ctx)
        out.append({"id": f"synth_qa_{i:03d}", "question": f"In which region was {person} born?",
                    "answer": region, "context": ctx})
    return out


def gsm8k(path, n=None, seed=0, exclude=()):
    rows = [json.loads(l) for l in open(path, encoding="utf8")]
    out = []
    for i, r in enumerate(rows):
        ans = r["answer"].split("####")[-1].strip().replace(",", "")
        if f"gsm8k_test_{i:04d}" in exclude:
            continue
        out.append({"id": f"gsm8k_test_{i:04d}", "question": r["question"], "answer": ans})
    if n:
        out = random.Random(seed).sample(out, n)
    return out


def hotpot(path, n=None, seed=0, types=("bridge",)):
    rows = json.load(open(path, encoding="utf8"))
    out = []
    for r in rows:
        if types and r.get("type") not in types:
            continue
        out.append({"id": r["_id"], "question": r["question"], "answer": r["answer"],
                    "context": [(t, s) for t, s in r["context"]], "type": r.get("type")})
    if n:
        out = random.Random(seed).sample(out, n)
    return out

"""LLM judge: a standard post-hoc attributor ("all-at-once", as in Who&When) run on the
injected traces already collected by run_pilot.py. No task is re-run.

For every decisive injection (U and M) in every condition, the judge reads the task, the
three agent messages (with element ids; with declared sources in the +prov conditions),
the wrong final answer and the correct answer, and names the agent and the element where
the decisive error was first introduced.

Examples
  python judge.py --list                                    # chat models on Together with prices
  python judge.py --results results/math_gsm8k_Qwen-Qwen3.5-9B.jsonl \
                  --data gsm8k:data/gsm8k_test.jsonl --model MODEL --dry-run
  python judge.py --results ... --data ... --model MODEL      # real run
  python judge.py --results ... --data ... --llm mock         # code check, free

Key and base URL from the environment only (OPENAI_API_KEY, OPENAI_BASE_URL).
"""
import argparse
import json
import os
import random
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

from r2.llm import RealLLM

CONDS = ["libera", "libera+prov", "strutt", "strutt+prov"]
KINDS = ["U", "M"]

SYSTEM = (
    "You are debugging a failed run of a multi-agent system. Three agents worked in sequence "
    "on the task: Agent 1, then Agent 2, then Agent 3. Each agent saw only the messages before it. "
    "The final answer is wrong. Exactly one decisive error was introduced at one point of the run; "
    "later agents may have copied it. Find the element where the decisive error FIRST appears.\n"
    "Every element has an id like M2.E3 (message 2 = Agent 2, element 3). M0 is the task. "
    "When agents declared sources, they are shown after the element; declared sources can be wrong.\n"
    'Answer with JSON only, on one line: {"agent": <1, 2 or 3>, "element": "<id>"}'
)


NUM_TOK = re.compile(r"\$?\d[\d,]*(?:\.\d+)?%?(?:/\d+)?|[+\-*/×÷=()]")


def strip_text(text):
    """Pre-registration 02: keep only numbers and arithmetic symbols, drop every word
    (the labels that say what a value is)."""
    toks = NUM_TOK.findall(text or "")
    out = " ".join(toks).strip()
    return out if any(ch.isdigit() for ch in out) else "(no number)"


def render_for_judge(messages, header0, strip=False):
    out = []
    for m, msg in enumerate(messages):
        out.append(f"--- {header0 if m == 0 else f'MESSAGE FROM AGENT {m}'} ---")
        for e in msg["elements"]:
            text = e["text"]
            if strip and m > 0:
                if e["id"].endswith(".R"):
                    continue  # reasoning paragraphs restate what values mean: removed
                text = strip_text(text)
            t = f"[{e['id']}] FINAL: {text}" if e.get("final") else f"[{e['id']}] {text}"
            if m > 0 and e.get("sources"):
                t += f"   (declared sources: {', '.join(e['sources'])})"
            out.append(t)
    return "\n".join(out)


def user_prompt(run, correct, header0, strip=False, no_answer=False):
    answer = "" if no_answer else f"\nCORRECT ANSWER: {correct}"
    return (render_for_judge(run["messages"], header0, strip)
            + f"\n\nFINAL ANSWER GIVEN: {run.get('final')}{answer}\n\n"
            + "Which element first introduced the decisive error? JSON only.")


def parse_judge(text):
    for m in reversed(list(re.finditer(r"\{[^{}]*\}", text or ""))):
        try:
            d = json.loads(m.group(0))
        except Exception:
            continue
        el = str(d.get("element", "")).strip().strip("[]")
        try:
            ag = int(d.get("agent"))
        except Exception:
            mm = re.match(r"M(\d+)\.", el)
            ag = int(mm.group(1)) if mm else None
        if el or ag:
            return ag, el
    mm = re.search(r"\bM([1-3])\.(E\d+|R|F)\b", text or "")
    if mm:
        return int(mm.group(1)), mm.group(0)
    return None, None


def load_answers(data):
    kind, path = data.split(":", 1)
    if kind == "gsm8k":
        out = {}
        for i, l in enumerate(open(path, encoding="utf8")):
            out[f"gsm8k_test_{i:04d}"] = json.loads(l)["answer"].split("####")[-1].strip().replace(",", "")
        return out, "TASK"
    if kind == "hotpot":
        return {r["_id"]: r["answer"] for r in json.load(open(path, encoding="utf8"))}, "CONTEXT AND QUESTION"
    raise SystemExit(f"unknown data {data}")


def together_models():
    import httpx  # installed with the openai package
    r = httpx.get(os.environ.get("OPENAI_BASE_URL", "https://api.together.xyz/v1").rstrip("/") + "/models",
                  headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
                           "User-Agent": "r2-judge/1.0", "Accept": "application/json"}, timeout=60)
    r.raise_for_status()
    d = r.json()
    return d if isinstance(d, list) else d.get("data", [])


def model_prices(model):
    for m in together_models():
        if m.get("id") == model:
            p = m.get("pricing") or {}
            return float(p.get("input") or 0), float(p.get("output") or 0)
    return 0.0, 0.0


class MockJudge:
    """Right with prob. 0.9 when sources are shown, 0.6 otherwise. Only to test the code."""

    def __init__(self, seed=3):
        self.rng = random.Random(seed)
        self.truth = {}

    def call(self, system, user, max_tokens=0):
        ag, el = self.truth[user]
        p = 0.9 if "declared sources" in user else 0.6
        if self.rng.random() > p:
            el = f"M{ag}.E{self.rng.randint(1, 4)}" if self.rng.random() < .5 else "M3.E1"
            ag = int(el[1])
        return json.dumps({"agent": ag, "element": el}), {"in": len(user) // 4, "out": 20}


def items(rows):
    for r in rows:
        if r.get("ok_major") is False:
            continue
        for k in KINDS:
            I = (r.get("inj") or {}).get(k) or {}
            if I.get("decisive") and I.get("run"):
                yield r, k, I


def boot_diff(pairs, B=2000, seed=1):
    if not pairs:
        return None
    rng = random.Random(seed)
    d = [a - b for a, b in pairs]
    m = sum(d) / len(d)
    bs = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(B))
    return m, bs[int(.025 * B)], bs[int(.975 * B) - 1]


def summarize(J):
    by = {}
    for j in J:
        by.setdefault((j["cond"], j["kind"]), []).append(j)
    L = ["GIUDICE LLM (attribuzione post-hoc, tutto in una volta)",
         f"{'cond':<14}{'tipo':>5}{'n':>5}{'agente':>9}{'elemento':>10}{'regola':>9}{'invalidi':>10}"]
    for c in CONDS:
        for k in KINDS:
            x = by.get((c, k), [])
            if not x:
                continue
            n = len(x)
            L.append(f"{c:<14}{k:>5}{n:>5}{sum(j['j_agent'] for j in x)/n:>9.3f}"
                     f"{sum(j['j_element'] for j in x)/n:>10.3f}{sum(j['rule_element'] for j in x)/n:>9.3f}"
                     f"{sum(j['invalid'] for j in x)/n:>10.3f}")
    L.append("\nEFFETTO DELLE FONTI SUL GIUDICE (elemento giusto: con fonti - senza; IC 95% bootstrap per compito)")
    for k in KINDS:
        for a, b, lab in (("libera+prov", "libera", "libera"), ("strutt+prov", "strutt", "strutt")):
            A = {j["task"]: j["j_element"] for j in by.get((a, k), [])}
            Bm = {j["task"]: j["j_element"] for j in by.get((b, k), [])}
            pairs = [(A[t], Bm[t]) for t in A if t in Bm]
            r = boot_diff(pairs)
            L.append(f"  {k} {lab:<8} " + ("n.d." if not r else f"{r[0]:+.3f}  [{r[1]:+.3f}, {r[2]:+.3f}]  (compiti appaiati {len(pairs)})"))
    L.append("\nEFFETTO DELLA STRUTTURA SUL GIUDICE (strutt - libera, senza fonti)")
    for k in KINDS:
        A = {j["task"]: j["j_element"] for j in by.get(("strutt", k), [])}
        Bm = {j["task"]: j["j_element"] for j in by.get(("libera", k), [])}
        pairs = [(A[t], Bm[t]) for t in A if t in Bm]
        r = boot_diff(pairs)
        L.append(f"  {k}          " + ("n.d." if not r else f"{r[0]:+.3f}  [{r[1]:+.3f}, {r[2]:+.3f}]  (compiti appaiati {len(pairs)})"))
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results")
    ap.add_argument("--data")
    ap.add_argument("--llm", choices=["openai", "mock"], default="openai")
    ap.add_argument("--model", default="mock")
    ap.add_argument("--max-cost", type=float, default=5.0)
    ap.add_argument("--max-tokens", type=int, default=1500)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--price-in", type=float, default=None)
    ap.add_argument("--price-out", type=float, default=None)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--strip", action="store_true", help="pre-registration 02: agent messages reduced to numbers")
    ap.add_argument("--no-answer", action="store_true", help="pre-registration 04: the correct answer is not shown")
    a = ap.parse_args()

    if a.list:
        ms = [m for m in together_models() if m.get("type") == "chat"]
        ms.sort(key=lambda m: float((m.get("pricing") or {}).get("output") or 0))
        for m in ms:
            p = m.get("pricing") or {}
            print(f"{m.get('id'):<60} in {p.get('input')}  out {p.get('output')}")
        return

    rows = [json.loads(l) for l in open(a.results)]
    answers, header0 = load_answers(a.data)
    todo = list(items(rows))
    out_path = a.results.replace(".jsonl", "") + f"_judge_{a.model.replace('/', '-')}" + ("_strip" if a.strip else "") + ("_noans" if a.no_answer else "") + ".jsonl"
    done = set()
    if os.path.exists(out_path):
        done = {(j["task"], j["cond"], j["kind"]) for j in map(json.loads, open(out_path))}
    todo = [(r, k, I) for r, k, I in todo if (r["task"], r["cond"], k) not in done]
    print(f"iniezioni decisive da giudicare: {len(todo)} (gia fatte: {len(done)}) | giudice {a.llm}:{a.model}")

    if a.llm == "mock":
        llm, pin, pout = MockJudge(), 0.0, 0.0
    else:
        pin, pout = (a.price_in, a.price_out) if a.price_in is not None else model_prices(a.model)
        if not (pin > 0 and pout > 0):
            raise SystemExit("Prezzi del modello non trovati: passa --price-in e --price-out (USD per milione di token).")
        print(f"prezzi {pin} / {pout} USD per milione di token | tetto {a.max_cost} USD")
        est_in = sum(len(user_prompt(I["run"], answers.get(r["task"]), header0, a.strip, a.no_answer)) / 4 + 250 for r, k, I in todo)
        est = (est_in * pin + len(todo) * a.max_tokens * pout) / 1e6
        print(f"stima massima: ~{est_in/1e6:.2f}M token in, costo <= ~{est:.2f} USD")
        if a.dry_run:
            return
        if est > a.max_cost:
            raise SystemExit("La stima supera il tetto: alza --max-cost o cambia modello.")
        llm = RealLLM("openai", a.model, temperature=0.0)

    lock = threading.Lock()
    spent = [0.0]

    def one(item):
        r, k, I = item
        u = user_prompt(I["run"], answers.get(r["task"]), header0, a.strip, a.no_answer)
        if a.llm == "mock":
            llm.truth[u] = (int(I["agent"]), I["element"])
        with lock:
            if spent[0] > a.max_cost:
                return None
        text, us = llm.call(SYSTEM, u, max_tokens=a.max_tokens)
        ag, el = parse_judge(text)
        j = {"task": r["task"], "cond": r["cond"], "kind": k, "true_agent": int(I["agent"]),
             "true_element": I["element"], "judge_agent": ag, "judge_element": el,
             "j_agent": ag == int(I["agent"]), "j_element": el == I["element"],
             "rule_element": bool(I.get("hit_element")), "invalid": ag is None and not el,
             "raw": text, "tokens_in": us["in"], "tokens_out": us["out"]}
        with lock:
            spent[0] += (us["in"] * pin + us["out"] * pout) / 1e6
            with open(out_path, "a") as f:
                f.write(json.dumps(j) + "\n")
        return j

    with ThreadPoolExecutor(max_workers=1 if a.llm == "mock" else a.workers) as ex:
        list(ex.map(one, todo))
    J = [json.loads(l) for l in open(out_path)]
    txt = summarize(J) + f"\n\nSPESA GIUDICE ~{spent[0]:.3f} USD (questa esecuzione)"
    print(txt)
    open(out_path.replace(".jsonl", "_report.txt"), "w").write(txt)


if __name__ == "__main__":
    sys.exit(main())

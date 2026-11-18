"""Pilot runner.

Examples
  python3 run_pilot.py --family math --data synth --llm mock            # code check, free
  python3 run_pilot.py --family qa   --data synth --llm mock
  python3 run_pilot.py --family math --data gsm8k:data/test.jsonl --llm openai --model MODEL --dry-run
  python3 run_pilot.py --family math --data gsm8k:data/test.jsonl --llm openai --model MODEL --n 5   # first real check
  python3 run_pilot.py --family qa   --data hotpot:data/hotpot_dev_distractor_v1.json --llm anthropic --model MODEL

Real runs need the key in the environment (never in files):
  read -s OPENAI_API_KEY && export OPENAI_API_KEY
and prices + max_cost_usd filled in config.json. Results are appended to
results/<family>_<model>.jsonl and the run resumes where it stopped.
"""
import argparse
import datetime
import hashlib
import json
import os
import sys

from r2.experiment import dump_manual_sample, report, run_pilot, summarize
from r2.families import FAMILIES
from r2.fmt import render_inputs
from r2.llm import MockMath, MockQA, RealLLM
from r2.prompts import prompts_hash
from r2 import tasks as T


def load_tasks(family, data, n, seed, cfg):
    if data == "synth":
        return (T.synth_math if family == "math" else T.synth_qa)(n, seed=seed)
    kind, path = data.split(":", 1)
    if kind == "gsm8k":
        ex = set()
        if cfg.get("exclude_ids_file") and os.path.exists(cfg["exclude_ids_file"]):
            ex = {l.strip() for l in open(cfg["exclude_ids_file"]) if l.strip()}
            print(f"esclusi {len(ex)} problemi gia usati")
        return T.gsm8k(path, n=n, seed=seed, exclude=ex)
    if kind == "hotpot":
        return T.hotpot(path, n=n, seed=seed, types=tuple(cfg.get("hotpot_types") or ()))
    raise SystemExit(f"unknown data {data}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["math", "qa"], required=True)
    ap.add_argument("--data", default="synth")
    ap.add_argument("--llm", choices=["mock", "openai", "anthropic"], default="mock")
    ap.add_argument("--model", default="mock")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--out", default="results")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()

    cfg = json.load(open(a.config))
    fam = FAMILIES[a.family]()
    n = a.n or cfg["n_tasks_per_family"]
    tasks = load_tasks(a.family, a.data, n, cfg["seed"], cfg)
    cfg_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]
    print(f"famiglia {a.family} | compiti {len(tasks)} | modello {a.llm}:{a.model}")
    print(f"hash istruzioni {prompts_hash(fam)} | hash config {cfg_hash}")

    if a.dry_run:
        k = len(cfg["kinds"])
        per_cond_max = cfg["reps"] * 3 + k * cfg["max_attempts"] * 2
        per_cond_typ = cfg["reps"] * 3 + k * 1.5 * 1.5
        ctx = sum(len(render_inputs([fam.task_message(t)], fam.header0)) for t in tasks) / len(tasks) / 4
        b = cfg["budget"]
        for label, per in (("tipico", per_cond_typ), ("massimo", per_cond_max)):
            calls = len(tasks) * 4 * per
            tin = calls * (ctx + 450)  # task input + system prompt + previous messages (rough)
            tout = calls * b["assumed_out_tokens_per_call"]
            cost = (tin * b["price_in_per_mtok"] + tout * b["price_out_per_mtok"]) / 1e6
            print(f"stima {label}: ~{calls:.0f} chiamate, ~{tin/1e6:.2f}M token in, ~{tout/1e6:.2f}M token out, "
                  f"costo ~{cost:.2f} USD" + ("" if b["price_in_per_mtok"] else " (prezzi non compilati)"))
        return

    if a.llm == "mock":
        llm = (MockMath if a.family == "math" else MockQA)(seed=7)
        budget = None
    else:
        b = cfg["budget"]
        if not (b["max_cost_usd"] > 0 and b["price_in_per_mtok"] > 0 and b["price_out_per_mtok"] > 0):
            raise SystemExit("Compila prezzi e max_cost_usd in config.json prima di una corsa vera.")
        llm = RealLLM(a.llm, a.model, temperature=cfg["temperature"], extra_body=cfg.get("extra_body"))
        budget = b

    os.makedirs(a.out, exist_ok=True)
    tag = f"{a.family}_{a.data.split(':')[0]}_{a.model}".replace("/", "-")
    out_path = os.path.join(a.out, f"{tag}.jsonl")
    with open(os.path.join(a.out, f"{tag}_info.json"), "w") as f:
        json.dump({"date": datetime.date.today().isoformat(), "family": a.family, "data": a.data,
                   "llm": a.llm, "model": a.model, "n": len(tasks), "prompts_hash": prompts_hash(fam),
                   "config_hash": cfg_hash, "config": cfg}, f, indent=2)
    workers = a.workers or (1 if a.llm == "mock" else cfg.get("workers", 8))
    print(f"chiamate in parallelo: {workers}")
    rows = run_pilot(tasks, llm, fam, cfg, out_path=out_path, budget=budget, workers=workers)
    S = summarize(rows, cfg)
    txt = report(S, rows, budget)
    print(txt)
    open(os.path.join(a.out, f"{tag}_report.txt"), "w").write(txt)
    dump_manual_sample(rows, os.path.join(a.out, f"{tag}_manual_check_20.txt"))


if __name__ == "__main__":
    main()

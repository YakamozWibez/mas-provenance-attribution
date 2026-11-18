"""Pilot runner (with checkpointing), metrics and gate."""
import itertools
import json
import os
import random

import numpy as np

from .pipeline import attribute, correct, false_alarm, inject, run_chain, source_fidelity

CONDS = [(s, p) for s in (False, True) for p in (False, True)]
CNAME = {(False, False): "libera", (False, True): "libera+prov",
         (True, False): "strutt", (True, True): "strutt+prov"}
Z_ALPHA, Z_BETA2 = 1.6449, 1.2816  # alpha = 0.05 (TOST), power 0.8


def n_required(psi, delta):
    return (Z_ALPHA + Z_BETA2) ** 2 * psi / delta ** 2


def _slim(run):
    return {"messages": run["messages"], "final": run["final"], "raw": run["raw"],
            "usage": run["usage"], "valid": run["valid"]}


def run_one(task, s, p, llm, fam, cfg, rng):
    vis = cfg.get("visibility", "spec")
    reps = [run_chain(task, s, p, llm, fam, max_tokens=cfg["max_tokens"], visibility=vis) for _ in range(cfg["reps"])]
    oks = [correct(r, task, fam) for r in reps]
    major = 2 * sum(oks) > len(oks)
    # base run for injection and for the rule checks: first correct repetition
    # when the majority is correct, otherwise the first repetition
    r1 = next(r for r, o in zip(reps, oks) if o) if major else reps[0]
    row = {"task": task["id"], "cond": CNAME[(s, p)], "s": s, "p": p, "ok": oks, "ok_major": major,
           "valid1": r1["valid"],
           "tokens_in": sum(r["usage"]["in"] for r in reps), "tokens_out": sum(r["usage"]["out"] for r in reps),
           "calls": sum(r["usage"]["calls"] for r in reps),
           "fid": source_fidelity(r1, fam) if p else None,
           "false_alarm": false_alarm(r1, p, fam) if major else None,
           "inj": {}, "clean": _slim(r1)}
    if major:
        for k in cfg["kinds"]:
            rec, att = inject(r1, task, s, p, llm, fam, rng, kind=k,
                              max_attempts=cfg["max_attempts"], max_tokens=cfg["max_tokens"], visibility=vis)
            row["calls"] += sum(a["run"]["usage"]["calls"] for a in att)
            row["tokens_in"] += sum(a["run"]["usage"]["in"] for a in att)
            row["tokens_out"] += sum(a["run"]["usage"]["out"] for a in att)
            info = {"attempts": len(att), "decisive": rec is not None,
                    "tried": [{"agent": x["agent"], "element": x["element"], "target": x["target"],
                               "value": x["value"], "final": x["run"]["final"], "decisive": x["decisive"]}
                              for x in att]}
            if rec:
                pa, pe, how = attribute(rec["run"], p, fam)
                info.update({"agent": rec["agent"], "element": rec["element"], "target": rec["target"],
                             "value": rec["value"], "pred_agent": pa, "pred_element": pe, "how": how,
                             "hit_agent": pa == rec["agent"], "hit_element": pe == rec["element"],
                             "run": _slim(rec["run"])})
            row["inj"][k] = info
    return row


def run_pilot(tasks, llm, fam, cfg, out_path=None, budget=None, log=print, workers=1):
    """Runs all (task, condition) pairs, `workers` at a time. Each finished pair is
    appended as one JSON line; already-done pairs are skipped (resume). Every pair
    has its own random generator, so the injected errors do not depend on order."""
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    done, rows = set(), []
    if out_path and os.path.exists(out_path):
        for l in open(out_path, encoding="utf8"):
            r = json.loads(l)
            rows.append(r); done.add((r["task"], r["cond"]))
    state = {"calls": sum(r["calls"] for r in rows),
             "cost": sum(_cost(r, budget) for r in rows) if budget else 0.0}
    jobs = [(t, s, p) for t in tasks for s, p in CONDS if (t["id"], CNAME[(s, p)]) not in done]
    left = {}
    for t, _, _ in jobs:
        left[t["id"]] = left.get(t["id"], 0) + 1
    lock, stop = threading.Lock(), threading.Event()

    def over():
        return budget and (state["calls"] >= budget["max_calls"]
                           or (budget["max_cost_usd"] > 0 and state["cost"] >= budget["max_cost_usd"]))

    def work(job):
        t, s, p = job
        if stop.is_set():
            return None
        rng = random.Random(f'{cfg["seed"]}|{t["id"]}|{CNAME[(s, p)]}')
        row = run_one(t, s, p, llm, fam, cfg, rng)
        with lock:
            rows.append(row)
            state["calls"] += row["calls"]
            state["cost"] += _cost(row, budget) if budget else 0.0
            if out_path:
                with open(out_path, "a", encoding="utf8") as f:
                    f.write(json.dumps(row) + "\n")
            left[t["id"]] -= 1
            if left[t["id"]] == 0:
                n_done = len({r["task"] for r in rows})
                log(f"{t['id']} finito ({n_done}/{len(tasks)} compiti): chiamate {state['calls']}"
                    + (f", costo ~{state['cost']:.2f} USD" if budget else ""))
            if over() and not stop.is_set():
                stop.set()
                log(f"STOP: budget raggiunto (chiamate {state['calls']}, costo ~{state['cost']:.2f} USD)")
        return row

    if workers <= 1:
        for job in jobs:
            if stop.is_set():
                break
            work(job)
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for f in as_completed([ex.submit(work, j) for j in jobs]):
                f.result()
    return rows


def _cost(row, budget):
    return (row["tokens_in"] * budget["price_in_per_mtok"] + row["tokens_out"] * budget["price_out_per_mtok"]) / 1e6


def _boot_diff(rows, cond_a, cond_b, kind, B=2000, seed=1):
    by = {}
    for r in rows:
        i = r["inj"].get(kind)
        if i and i.get("decisive"):
            by.setdefault(r["task"], {})[r["cond"]] = i["hit_element"]
    tasks = list(by)
    rng = np.random.default_rng(seed)

    def stat(ts):
        a = [by[t][cond_a] for t in ts if cond_a in by[t]]
        b = [by[t][cond_b] for t in ts if cond_b in by[t]]
        return (np.mean(a) if a else np.nan) - (np.mean(b) if b else np.nan)
    if not tasks:
        return np.nan, np.nan, np.nan
    boots = [stat(rng.choice(tasks, len(tasks))) for _ in range(B)]
    return stat(tasks), np.nanpercentile(boots, 2.5), np.nanpercentile(boots, 97.5)


def _major(r):
    return r.get("ok_major", r["ok"][0])


def _pairflip(oks):
    pairs = list(itertools.combinations(oks, 2))
    return np.mean([a != b for a, b in pairs]) if pairs else np.nan


def _pos(eid):
    from .fmt import elem_index, msg_index
    return (msg_index(eid), elem_index(eid))


def summarize(rows, cfg, budget_tasks=None):
    g_cfg = cfg["gate"]
    out = {"per_cond": {}, "psi": {}, "gate": {}}
    for c in CNAME.values():
        rs = [r for r in rows if r["cond"] == c]
        if not rs:
            continue
        d = {"n": len(rs), "P": np.mean([_major(r) for r in rs]),
             "invalid": 1 - np.mean([r["valid1"] for r in rs]),
             # instability: share of pairs of repetitions with a different outcome
             "retest_flip": np.mean([_pairflip(r["ok"]) for r in rs]) if cfg["reps"] > 1 else np.nan,
             "tokens_per_run": np.mean([r["clean"]["usage"]["in"] + r["clean"]["usage"]["out"] for r in rs])}
        fa = [r["false_alarm"] for r in rs if r["false_alarm"] is not None]
        d["false_alarm"] = np.mean(fa) if fa else np.nan
        for k in cfg["kinds"]:
            inj = [r["inj"][k] for r in rs if k in r["inj"]]
            dec = [i for i in inj if i["decisive"]]
            att = sum(i["attempts"] for i in inj)
            d[f"yield_{k}"] = len(dec) / att if att else np.nan
            d[f"A_{k}"] = np.mean([i["hit_element"] for i in dec]) if dec else np.nan
            d[f"Aagent_{k}"] = np.mean([i["hit_agent"] for i in dec]) if dec else np.nan
            d[f"n_{k}"] = len(dec)
        decs = [i for r in rs for i in r["inj"].values() if i.get("decisive")]
        early = [i for i in decs if not i["hit_element"] and _pos(i["pred_element"]) < _pos(i["element"])]
        d["early_miss"] = len(early) / len(decs) if decs else np.nan
        fid = [r["fid"] for r in rs if r["fid"]]
        if fid:
            el = sum(f["elements"] for f in fid); cov = sum(f["covered"] for f in fid)
            d["coverage"] = cov / el if el else np.nan
            d["fidelity"] = sum(f["faithful"] for f in fid) / cov if cov else np.nan
            d["fid_partial"] = sum(f.get("partial", 0) for f in fid) / cov if cov else np.nan
        out["per_cond"][c] = d
    by = {}
    for r in rows:
        by.setdefault(r["task"], {})[r["cond"]] = _major(r)
    full = [t for t in by if len(by[t]) == 4]
    for a, b in itertools.combinations(CNAME.values(), 2):
        psi = np.mean([by[t][a] != by[t][b] for t in full]) if full else np.nan
        out["psi"][f"{a} vs {b}"] = (psi, n_required(psi, cfg["delta"]))
    out["utility"] = {f"{k} {st}": _boot_diff(rows, f"{st}+prov", st, k)
                      for k in cfg["kinds"] for st in ("libera", "strutt")}
    pc = out["per_cond"]
    G = out["gate"]
    base = pc.get("libera", {}).get("P", np.nan)
    G[f"1 niente soffitto/pavimento ({g_cfg['p_min']}<=P_base<={g_cfg['p_max']})"] = bool(g_cfg["p_min"] <= base <= g_cfg["p_max"])
    ys = [pc[c][f"yield_{k}"] for c in pc for k in cfg["kinds"] if not np.isnan(pc[c][f"yield_{k}"])]
    G[f"2 errori decisivi (resa >= {g_cfg['yield_min']})"] = bool(ys) and all(y >= g_cfg["yield_min"] for y in ys)
    covs = [pc[c]["coverage"] for c in pc if "coverage" in pc[c]]
    G[f"3 copertura fonti >= {g_cfg['coverage_min']}"] = bool(covs) and all(x >= g_cfg["coverage_min"] for x in covs)
    # (pilot 2) Rule validity = how often a false alarm BEFORE the injected point makes
    # the rule blame the wrong place. Clean-run false alarms are reported, not gated.
    ems = [pc[c]["early_miss"] for c in pc if not np.isnan(pc[c].get("early_miss", np.nan))]
    G[f"4a regola: attribuzioni sbagliate per falso allarme prima del punto iniettato <= {g_cfg['early_miss_max']}"] = \
        bool(ems) and all(x <= g_cfg["early_miss_max"] for x in ems)
    G["4b regola: controllo a mano su 20 casi"] = "DA FARE A MANO (manual_check_20.txt)"
    if cfg["reps"] > 1:
        flip = pc.get("libera", {}).get("retest_flip", np.nan)
        G[f"5 stabilita della condizione base (cambi tra ripetizioni <= {g_cfg['retest_max']})"] = bool(flip <= g_cfg["retest_max"])
    psis = [v[0] for v in out["psi"].values() if not np.isnan(v[0])]
    if psis:
        n_need = n_required(max(psis), cfg["delta"])
        G["6 fattibilita"] = (f"servono ~{n_need:.0f} compiti per famiglia (psi max {max(psis):.3f}); "
                              + ("budget in compiti non fissato" if budget_tasks is None
                                 else ("OK" if n_need <= budget_tasks else "NON OK")))
    G[f"(validita) output invalidi < {g_cfg['invalid_max']}"] = all(pc[c]["invalid"] < g_cfg["invalid_max"] for c in pc)
    return out


def report(S, rows, budget=None):
    lines = []
    hdr = ["n", "P", "invalid", "retest_flip", "false_alarm", "early_miss", "tokens_per_run",
           "yield_U", "A_U", "n_U", "yield_M", "A_M", "n_M", "coverage", "fidelity", "fid_partial"]
    lines.append("PER CONDIZIONE")
    lines.append("cond".ljust(13) + "".join(h[:10].rjust(11) for h in hdr))
    for c, d in S["per_cond"].items():
        cells = []
        for h in hdr:
            v = d.get(h, "-")
            cells.append((f"{v:.3f}" if isinstance(v, (float, np.floating)) else str(v)).rjust(11))
        lines.append(c.ljust(13) + "".join(cells))
    lines.append("\nUTILITA DELLE FONTI (A con fonti - A senza; IC 95% bootstrap per compito)")
    for k, (e, lo, hi) in S["utility"].items():
        lines.append(f"  {k:12s} {e:+.3f}  [{lo:+.3f}, {hi:+.3f}]")
    lines.append("\nPSI E COMPITI NECESSARI")
    for k, (psi, n) in S["psi"].items():
        lines.append(f"  {k:28s} psi={psi:.3f}  n~{n:.0f}")
    lines.append("\nGATE")
    for k, v in S["gate"].items():
        lines.append(f"  {k}: {v}")
    calls = sum(r["calls"] for r in rows)
    tin = sum(r["tokens_in"] for r in rows); tout = sum(r["tokens_out"] for r in rows)
    lines.append(f"\nCHIAMATE {calls} | TOKEN IN {tin} | TOKEN OUT {tout}"
                 + (f" | COSTO ~{(tin*budget['price_in_per_mtok']+tout*budget['price_out_per_mtok'])/1e6:.2f} USD" if budget else ""))
    return "\n".join(lines)


def dump_manual_sample(rows, path, n=20, seed=3):
    rng = random.Random(seed)
    pool = [(r, k) for r in rows for k, i in r["inj"].items() if i.get("decisive")]
    rng.shuffle(pool)
    with open(path, "w", encoding="utf8") as f:
        f.write("Per ogni caso: segui a mano da dove nasce l'errore e scrivi se la regola ha ragione (SI/NO).\n\n")
        for r, k in pool[:n]:
            i = r["inj"][k]
            f.write(f"=== {r['task']} | {r['cond']} | tipo {k} | iniettato {i['element']} "
                    f"({i['target']} -> {i['value']}) | regola -> {i['pred_element']} ({i['how']}) | giusto? ___\n")
            for msg in i["run"]["messages"]:
                for e in msg["elements"]:
                    src = f"  <- {', '.join(e['sources'])}" if e["sources"] else ""
                    f.write(f"  [{e['id']}] {e['text']}{src}\n")
            f.write("\n")

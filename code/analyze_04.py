"""Analysis fixed by pre-registration 04 (prereg/04_replication_kstar_noanswer.md).
Committed before any of the data it analyses exist.

Step 1, prediction (only the agents' traces, no judge):
  python ambiguity.py      RESULTS.jsonl > AMB.jsonl
  python ambiguity_star.py RESULTS.jsonl > AMBSTAR.jsonl
  python analyze_04.py predict AMB.jsonl AMBSTAR.jsonl
Step 2, test (after the four judge runs):
  python analyze_04.py test RESULTS.jsonl AMB.jsonl AMBSTAR.jsonl \
         JUDGE_FULL.jsonl JUDGE_STRIP.jsonl JUDGE_FULL_NOANS.jsonl JUDGE_STRIP_NOANS.jsonl
"""
import json
import random
import sys

B, SEED, TOL, MIN_GROUP = 5000, 4, 0.20, 5
STRUCTS = ("libera", "strutt")


def load(p):
    return [json.loads(l) for l in open(p)]


def key(r, kind=None):
    return (r["task"], r["cond"], kind or r["kind"])


def mean(x):
    x = list(x)
    return sum(x) / len(x) if x else float("nan")


def boot(units, stat):
    rng = random.Random(SEED)
    est = stat(units)
    bs = sorted(stat([rng.choice(units) for _ in units]) for _ in range(B))
    return est, bs[int(.025 * B)], bs[int(.975 * B) - 1]


def fmt(t):
    return f"{t[0]:+.3f} [{t[1]:+.3f}, {t[2]:+.3f}]"


def bound_k(a, view="strip"):
    return 1 / a["k_" + view]


def bound_kstarB(s):
    """Ideal reader trusting the order cue: 0 if the cue clears the culprit, else 1/k*_B."""
    return 0.0 if s["culprit_cleared_B"] else 1 / s["k_starB"]


def pairs(kind, A, S, J=None):
    """Per task and structure: dict with bounds with/without sources and judge hits."""
    out = []
    for (t, c, k), a in A.items():
        if k != kind or not c.endswith("+prov"):
            continue
        s = c[:-5]
        if (t, s, k) not in A:
            continue
        row = {"task": t, "struct": s,
               "k_with": bound_k(a), "k_without": bound_k(A[(t, s, k)])}
        if S is not None and (t, c, k) in S and (t, s, k) in S:
            row["ks_with"], row["ks_without"] = bound_kstarB(S[(t, c, k)]), bound_kstarB(S[(t, s, k)])
        if J is not None:
            if (t, c, k) not in J or (t, s, k) not in J:
                continue
            row["J_with"], row["J_without"] = int(J[(t, c, k)]), int(J[(t, s, k)])
        out.append(row)
    return out


def verdict_equiv(D):
    return "SUPPORTED" if -TOL <= D[1] and D[2] <= TOL else ("CONTRADICTED" if D[1] > 0 or D[2] < 0 else "INCONCLUSIVE")


def predict(amb_p, star_p):
    A = {key(r): r for r in load(amb_p)}
    S = {key(r): r for r in load(star_p)}
    for kind in ("M", "U"):
        P = pairs(kind, A, S)
        print(f"=== {kind}: pairs {len(P)} (stripped view, structures pooled)")
        print(f"  predicted effect of sources from k   : {mean(p['k_with'] - p['k_without'] for p in P):+.3f}")
        Q = [p for p in P if "ks_with" in p]
        print(f"  predicted effect of sources from k*_B: {mean(p['ks_with'] - p['ks_without'] for p in Q):+.3f}")
        nos = [S[k] for k in S if k[2] == kind and k[1] in STRUCTS]
        print(f"  no sources: k>=2 in {sum(s['k'] >= 2 for s in nos)}/{len(nos)}; "
              f"order cue clears culprit in {sum(s['culprit_cleared_B'] for s in nos)}/{len(nos)}")


def test(res_p, amb_p, star_p, jf_p, js_p, jfn_p, jsn_p):
    A = {key(r): r for r in load(amb_p)}
    S = {key(r): r for r in load(star_p)}
    JF, JS, JFN, JSN = ({key(j): j["j_element"] for j in load(p)} for p in (jf_p, js_p, jfn_p, jsn_p))
    kind = "M"
    P = pairs(kind, A, S, JS)
    print(f"=== PRIMARY (M errors, stripped view, correct answer shown, pairs {len(P)}) ===")
    h1 = boot(P, lambda u: mean(p["J_with"] - p["J_without"] for p in u))
    print(f"H1  effect of sources (replication)        {fmt(h1)}  -> {'REPLICATED' if h1[1] > 0 else 'NOT replicated'}")
    d2 = boot(P, lambda u: mean((p["J_with"] - p["J_without"]) - (p["k_with"] - p["k_without"]) for p in u))
    print(f"H2  observed - predicted from k            {fmt(d2)}  -> {verdict_equiv(d2)}")
    Q = [p for p in P if "ks_with" in p]
    d3 = boot(Q, lambda u: mean((p["J_with"] - p["J_without"]) - (p["ks_with"] - p["ks_without"]) for p in u))
    v3 = ("A (informational): k* predicts the effect" if -TOL <= d3[1] and d3[2] <= TOL else
          "B (accessibility): sources help beyond what k* predicts" if d3[1] > 0 else
          "neither: sources help less than k* predicts" if d3[2] < 0 else "INCONCLUSIVE")
    print(f"H3  observed - predicted from k*_B         {fmt(d3)}  -> {v3}")

    print("\n=== SECONDARY ===")
    # H4: among no-source stripped cases with k>=2, does the order cue resolving the case help the judge?
    cases = [s for k, s in S.items() if k[2] == kind and k[1] in STRUCTS and s["k"] >= 2 and k in JS]
    g1 = [JS[key(s)] for s in cases if s["k_starB"] == 1 and not s["culprit_cleared_B"]]
    g2 = [JS[key(s)] for s in cases if not (s["k_starB"] == 1 and not s["culprit_cleared_B"])]
    if len(g1) >= MIN_GROUP and len(g2) >= MIN_GROUP:
        units = [(1, v) for v in g1] + [(2, v) for v in g2]
        h4 = boot(units, lambda u: mean(v for g, v in u if g == 1) - mean(v for g, v in u if g == 2))
        print(f"H4  J(order resolves) - J(order does not)  {fmt(h4)}  (n {len(g1)} vs {len(g2)})")
    else:
        print(f"H4  not testable (n {len(g1)} vs {len(g2)}, need >= {MIN_GROUP} each)")
    # H5: no correct answer
    PN = pairs(kind, A, S, JSN)
    common = {(p["task"], p["struct"]): p for p in PN}
    both = [(p, common[(p["task"], p["struct"])]) for p in P if (p["task"], p["struct"]) in common]
    h5a = boot([p for _, p in both], lambda u: mean(p["J_with"] - p["J_without"] for p in u))
    h5 = boot(both, lambda u: mean((q["J_with"] - q["J_without"]) - (p["J_with"] - p["J_without"]) for p, q in u))
    nos_keys = [k for k in JS if k[2] == kind and k[1] in STRUCTS and k in JSN]
    drop = boot(nos_keys, lambda u: mean(JSN[k] - JS[k] for k in u))
    print(f"H5a no answer: J change, stripped, no sources   {fmt(drop)}")
    print(f"H5b no answer: effect of sources (stripped)     {fmt(h5a)}")
    print(f"H5  effect without answer - with answer         {fmt(h5)}  -> "
          f"{'SUPPORTED (sources matter more without the answer)' if h5[1] > 0 else 'not supported'}")
    PFN = pairs(kind, A, S, JFN)
    if PFN:
        print(f"    desc. effect of sources, full traces, no answer {fmt(boot(PFN, lambda u: mean(p['J_with'] - p['J_without'] for p in u)))}")
    # H6: performance cost
    R = {(r["task"], r["cond"]): int(bool(r.get("ok_major"))) for r in load(res_p)}
    tasks = sorted({t for t, _ in R})
    d = [R[(t, "strutt+prov")] - R[(t, "libera")] for t in tasks if (t, "strutt+prov") in R and (t, "libera") in R]
    h6 = boot(d, lambda u: mean(u))
    print(f"H6  P(structured + sources) - P(free)           {fmt(h6)}  (n {len(d)}) -> "
          f"{'SUPPORTED (cost)' if h6[2] < 0 else 'not supported'}")
    for c in ("libera+prov", "strutt"):
        dd = [R[(t, c)] - R[(t, "libera")] for t in tasks if (t, c) in R and (t, "libera") in R]
        print(f"    desc. P({c}) - P(free)  {fmt(boot(dd, lambda u: mean(u)))}")
    print("\n=== DESCRIPTIVE ===")
    PF = pairs(kind, A, S, JF)
    print(f"effect of sources, full traces, answer shown   {fmt(boot(PF, lambda u: mean(p['J_with'] - p['J_without'] for p in u)))}")
    PU = pairs("U", A, S, JS)
    if PU:
        print(f"U errors: effect of sources, stripped          {fmt(boot(PU, lambda u: mean(p['J_with'] - p['J_without'] for p in u)))}")


if __name__ == "__main__":
    if sys.argv[1] == "predict":
        predict(sys.argv[2], sys.argv[3])
    else:
        test(*sys.argv[2:9])

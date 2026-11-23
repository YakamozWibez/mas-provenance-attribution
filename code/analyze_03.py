"""Analysis fixed by pre-registration 03 (prereg/03_ambiguity_bound.md). Committed before
any of the data it analyses exist.

Step 1, prediction (uses only the agents' traces, no judge):
  python ambiguity.py RESULTS.jsonl > AMB.jsonl
  python analyze_03.py predict AMB.jsonl
Step 2, test (after the judge runs):
  python analyze_03.py test AMB.jsonl JUDGE_FULL.jsonl JUDGE_STRIP.jsonl
"""
import json
import random
import sys

B, SEED, TOL, MIN_GROUP = 5000, 3, 0.20, 5
STRUCTS = ("libera", "strutt")


def load(p):
    return [json.loads(l) for l in open(p)]


def mean(x):
    return sum(x) / len(x) if x else float("nan")


def boot(units, stat):
    rng = random.Random(SEED)
    est = stat(units)
    bs = sorted(stat([rng.choice(units) for _ in units]) for _ in range(B))
    return est, bs[int(.025 * B)], bs[int(.975 * B) - 1]


def fmt(t):
    return f"{t[0]:+.3f} [{t[1]:+.3f}, {t[2]:+.3f}]"


def index(rows, key):
    return {(r["task"], r["cond"], r["kind"]): r[key] for r in rows}


def pairs(A, kind, view, J=None):
    """Per task and structure: (task, struct, inv_k_with, inv_k_without, J_with, J_without)."""
    kk = "k_" + view
    a = {(r["task"], r["cond"], r["kind"]): r for r in A}
    out = []
    for (t, c, k), r in a.items():
        if k != kind or not c.endswith("+prov"):
            continue
        s = c[:-5]
        o = a.get((t, s, k))
        if not o:
            continue
        row = [t, s, 1 / r[kk], 1 / o[kk]]
        if J is not None:
            if (t, c, k) not in J or (t, s, k) not in J:
                continue
            row += [int(J[(t, c, k)]), int(J[(t, s, k)])]
        out.append(row)
    return out


def predict(amb_p):
    A = load(amb_p)
    print(f"cases: {len(A)}")
    for kind in ("M", "U"):
        print(f"\n=== {kind} ===")
        for view in ("full", "strip"):
            for c in ("libera", "libera+prov", "strutt", "strutt+prov"):
                ks = [r["k_" + view] for r in A if r["kind"] == kind and r["cond"] == c]
                if ks:
                    print(f"  bound 1/k  {view:<5} {c:<12} n={len(ks):<4} {mean([1/k for k in ks]):.3f}")
            P = pairs(A, kind, view)
            if P:
                print(f"  PREDICTED effect of sources, {view}: {mean([p[2]-p[3] for p in P]):+.3f} (pairs {len(P)})")


def test(amb_p, full_p, strip_p):
    A = load(amb_p)
    JF, JS = index(load(full_p), "j_element"), index(load(strip_p), "j_element")
    for kind in ("M", "U"):
        print(f"\n=== {kind} ===")
        P = pairs(A, kind, "strip", JS)
        if not P:
            print("  no pairs")
            continue
        d_obs = boot(P, lambda u: mean([p[4] - p[5] for p in u]))
        d_pred = mean([p[2] - p[3] for p in P])
        D = boot(P, lambda u: mean([(p[4] - p[5]) - (p[2] - p[3]) for p in u]))
        print(f"  pairs (stripped, both structures pooled): {len(P)}")
        print(f"  H2b observed effect of sources   {fmt(d_obs)}")
        print(f"      predicted from k             {d_pred:+.3f}")
        print(f"  H2  observed - predicted         {fmt(D)}")
        if kind == "M":
            v2 = ("SUPPORTED" if -TOL <= D[1] and D[2] <= TOL else
                  "CONTRADICTED" if D[1] > 0 or D[2] < 0 else "INCONCLUSIVE")
            print(f"      -> H2b {'REPLICATED' if d_obs[1] > 0 else 'NOT replicated'}; H2 {v2}")
        # H1: bound in stripped view without sources
        cases = [r for r in A if r["kind"] == kind and r["cond"] in STRUCTS
                 and (r["task"], r["cond"], kind) in JS]
        h1 = boot(cases, lambda u: mean([JS[(r["task"], r["cond"], kind)] - 1 / r["k_strip"] for r in u]))
        print(f"  H1  J - bound (stripped, no sources)  {fmt(h1)}  (n={len(cases)}; "
              f"J={mean([JS[(r['task'], r['cond'], kind)] for r in cases]):.3f}, "
              f"bound={mean([1/r['k_strip'] for r in cases]):.3f})")
        g1 = [JS[(r["task"], r["cond"], kind)] for r in cases if r["k_strip"] == 1]
        g2 = [JS[(r["task"], r["cond"], kind)] for r in cases if r["k_strip"] >= 2]
        if len(g1) >= MIN_GROUP and len(g2) >= MIN_GROUP:
            units = [(1, v) for v in g1] + [(2, v) for v in g2]
            h3 = boot(units, lambda u: mean([v for g, v in u if g == 1] or [0]) - mean([v for g, v in u if g == 2] or [0]))
            print(f"  H3  J(k=1) - J(k>=2)             {fmt(h3)}  (n {len(g1)} vs {len(g2)})")
        else:
            print(f"  H3  not testable (n {len(g1)} vs {len(g2)}, need >= {MIN_GROUP} each)")
        # descriptive
        for c in ("libera+prov", "strutt+prov"):
            x = [(JS[(r["task"], c, kind)], 1 / r["k_strip"]) for r in A
                 if r["kind"] == kind and r["cond"] == c and (r["task"], c, kind) in JS]
            if x:
                print(f"  desc. stripped {c:<12} J={mean([a for a,_ in x]):.3f}  bound={mean([b for _,b in x]):.3f}")
        PF = pairs(A, kind, "full", JF)
        if PF:
            print(f"  desc. effect of sources, full traces {fmt(boot(PF, lambda u: mean([p[4]-p[5] for p in u])))}")
    print(f"\nDecision rule (for M, see prereg/03):")
    print(f"  H2b: lower bound > 0 -> pre-registration 02 replicated")
    print(f"  H2 : interval inside [-{TOL}, +{TOL}] -> SUPPORTED; interval excludes 0 -> CONTRADICTED; else INCONCLUSIVE")
    print(f"  H1 : lower bound > 0 -> bound VIOLATED (the trace carries information k ignores)")
    print(f"  H3 : lower bound > 0 -> ambiguity predicts which cases are missed")


if __name__ == "__main__":
    if sys.argv[1] == "predict":
        predict(sys.argv[2])
    else:
        test(sys.argv[2], sys.argv[3], sys.argv[4])

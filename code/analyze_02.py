"""Analysis fixed by pre-registration 02 (prereg/02_non_injective.md). Written and committed
before the stripped judge run.

  python analyze_02.py FULL_JUDGE.jsonl STRIP_JUDGE.jsonl
"""
import json
import random
import sys

B, SEED = 5000, 2


def load(p):
    return {(j["task"], j["cond"], j["kind"]): j["j_element"] for j in map(json.loads, open(p))}


def pairs(D, kind):
    """(with sources, without sources) per task, within the same structure, pooled."""
    out = []
    for struct in ("libera", "strutt"):
        for (t, c, k), v in D.items():
            if k == kind and c == struct + "+prov" and (t, struct, kind) in D:
                out.append((t, struct, int(v), int(D[(t, struct, kind)])))
    return out


def boot(units, stat):
    rng = random.Random(SEED)
    est = stat(units)
    bs = sorted(stat([rng.choice(units) for _ in units]) for _ in range(B))
    return est, bs[int(.025 * B)], bs[int(.975 * B) - 1]


def mean(x):
    return sum(x) / len(x) if x else float("nan")


def main(full_p, strip_p):
    F, S = load(full_p), load(strip_p)
    print(f"cases: full {len(F)}, stripped {len(S)}")
    for kind in ("M", "U"):
        noprov_keys = [key for key in F if key[2] == kind and not key[1].endswith("+prov") and key in S]
        h1 = boot(noprov_keys, lambda ks: mean([S[k] for k in ks]) - mean([F[k] for k in ks]))
        pf, ps = pairs(F, kind), pairs(S, kind)
        h2 = boot(ps, lambda u: mean([a - b for _, _, a, b in u])) if ps else None
        dfull = boot(pf, lambda u: mean([a - b for _, _, a, b in u])) if pf else None
        # interaction: tasks paired in both runs
        both = [(p, q) for p in ps for q in pf if p[:2] == q[:2]]
        h3 = boot(both, lambda u: mean([(p[2] - p[3]) - (q[2] - q[3]) for p, q in u])) if both else None
        print(f"\n=== {kind} ===")
        print(f"J without sources: full {mean([F[k] for k in noprov_keys]):.3f}, "
              f"stripped {mean([S[k] for k in noprov_keys]):.3f} (n={len(noprov_keys)})")
        print(f"H1 drop from stripping (no sources)  {h1[0]:+.3f} [{h1[1]:+.3f}, {h1[2]:+.3f}]")
        if dfull:
            print(f"   effect of sources, full traces     {dfull[0]:+.3f} [{dfull[1]:+.3f}, {dfull[2]:+.3f}] (pairs {len(pf)})")
        if h2:
            print(f"H2 effect of sources, stripped       {h2[0]:+.3f} [{h2[1]:+.3f}, {h2[2]:+.3f}] (pairs {len(ps)})")
        if h3:
            print(f"H3 interaction (stripped - full)     {h3[0]:+.3f} [{h3[1]:+.3f}, {h3[2]:+.3f}] (pairs {len(both)})")
    print("\nDecision rule (for M, see prereg/02):")
    print("  H1 needs: drop <= -0.20 AND upper bound < 0; otherwise INCONCLUSIVE")
    print("  H1 ok and H2 lower bound > 0 -> prediction SUPPORTED")
    print("  H1 ok and H2 lower bound <= 0 -> prediction NOT SUPPORTED in this setting")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

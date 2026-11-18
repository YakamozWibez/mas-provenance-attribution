"""Chain execution, fault injection and deterministic attribution (family-generic)."""
import copy
import re

from .fmt import elem_index, msg_index, parse_output, render_inputs
from .prompts import system_prompt

N_AGENTS = 3


def run_chain(task, structured, provenance, llm, fam, messages=None, start=1, max_tokens=600, visibility="spec"):
    """Run agents start..3. `messages` holds messages 0..start-1 (kept fixed).
    visibility "spec": only agent 1 sees the task; later agents see the agents'
    messages only, so an error is not silently repaired by re-reading the task.
    visibility "full": every agent sees everything before it."""
    if messages is None:
        messages = [fam.task_message(task)]
    messages = copy.deepcopy(messages[:start])
    usage = {"in": 0, "out": 0, "calls": 0}
    raw = []
    for a in range(start, N_AGENTS + 1):
        text, u = llm.call(system_prompt(fam, a, structured, provenance),
                           render_inputs(messages, fam.header0, skip_task=(visibility == "spec" and a >= 2)),
                           max_tokens=max_tokens)
        usage["in"] += u["in"]; usage["out"] += u["out"]; usage["calls"] += 1
        raw.append(text)
        messages.append(parse_output(text, structured, a))
    final = messages[-1]["final"]
    return {"messages": messages, "final": final, "usage": usage, "raw": raw,
            "valid": all(m["valid"] for m in messages[1:]) and final is not None}


def correct(run, task, fam):
    return run["valid"] and fam.answer_ok(run["final"], task)


# ---------------------------------------------------------------- injection
def inject(clean, task, structured, provenance, llm, fam, rng, kind="U", max_attempts=3, max_tokens=600,
           visibility="spec"):
    """kind U: replace a token with a FABRICATED value that appears nowhere.
    kind M: misbinding - replace it with ANOTHER value present in the task input.
    Keep the first attempt that is decisive (correct clean run -> wrong answer)."""
    msgs = clean["messages"]
    used = fam.used_index(task, msgs)
    pool = fam.misbind_pool(task)
    items1 = [e for e in msgs[1]["elements"] if not e.get("final") and elem_index(e["id"]) >= 1
              and fam.replaceable(e["text"])]
    items2 = [e for e in msgs[2]["elements"] if not e.get("final") and elem_index(e["id"]) >= 1
              and fam.replaceable(e["text"])]
    # agent 2: only its last step, whose result is what the next agent uses
    cands = [(1, e) for e in items1] + ([(2, items2[-1])] if items2 else [])
    rng.shuffle(cands)
    attempts = []
    for a, e in cands[:max_attempts]:
        target = fam.replaceable(e["text"])[-1] if a == 2 else rng.choice(fam.replaceable(e["text"]))
        if kind == "U":
            v = fam.fabricate(target, used, rng)
        else:
            # External misbinding: a TRUE value from ANOTHER sentence of the task input,
            # which the element does not legitimately receive: absent from earlier agent
            # messages, from the agent's own earlier items, from the element itself, and
            # from the task sentence(s) that hold the original value. The rule does not
            # use declared sources, so the injected errors are the same in all conditions.
            own = elem_index(e["id"])
            excluded = set(fam.tokens(e["text"]))
            # (v3) values passed on by EARLIER agents are allowed: whether the swap is
            # detectable then depends on how precise the declared sources are. Values in
            # the agent's own earlier items stay excluded: the rule always allows them.
            for x in msgs[a]["elements"]:
                if elem_index(x["id"]) < own:
                    excluded |= fam.tokens(x["text"])
            ttoks = fam.tokens(target)
            m0 = msgs[0]["elements"]
            home = [u for u in m0 if fam.tokens(u["text"]) & ttoks]
            home_toks = set().union(*[fam.tokens(u["text"]) for u in home]) if home else set()
            opts = []
            for u in m0:
                if u in home:
                    continue
                for x in fam.replaceable(u["text"]):
                    tx = fam.tokens(x)
                    if tx and not (tx & excluded) and not (tx & home_toks) and not fam.same(x, target):
                        opts.append(x)
            v = rng.choice(opts) if opts else None
        if v is None:
            continue
        new_text = re.sub(rf"(?<![\w.]){re.escape(target)}(?![\w])", v, e["text"], count=1)
        if new_text == e["text"]:
            continue
        pert = copy.deepcopy(msgs[: a + 1])
        for pe in pert[a]["elements"]:
            if pe["id"] == e["id"]:
                pe["text"] = new_text
        run = run_chain(task, structured, provenance, llm, fam, messages=pert, start=a + 1,
                        max_tokens=max_tokens, visibility=visibility)
        rec = {"kind": kind, "agent": a, "element": e["id"], "target": target, "value": v,
               "run": run, "decisive": not correct(run, task, fam)}
        attempts.append(rec)
        if rec["decisive"]:
            return rec, attempts
    return None, attempts


# ---------------------------------------------------------------- attribution
def allowed_texts(el, msgs, provenance):
    """(text, is_task) pairs an element may legitimately draw on. With
    provenance and declared sources: only those. Otherwise everything before it."""
    m = msg_index(el["id"])
    own = elem_index(el["id"])
    if provenance and el["sources"]:
        # Declared sources across messages + the agent's own earlier items
        # (provenance is about what flows BETWEEN agents).
        out = []
        for s in el["sources"]:
            k = msg_index(s)
            if k < len(msgs):
                out += [(x["text"], k == 0) for x in msgs[k]["elements"] if x["id"] == s]
        out += [(x["text"], False) for x in msgs[m]["elements"] if elem_index(x["id"]) < own]
        return out
    out = [(x["text"], k == 0) for k in range(m) for x in msgs[k]["elements"]]
    out += [(x["text"], False) for x in msgs[m]["elements"] if elem_index(x["id"]) < own]
    return out


def supported(el, msgs, provenance, fam):
    ok, computed = fam.internal(el["text"])
    if not ok:
        return False
    toks = fam.tokens(el["text"]) - computed
    return fam.covered(toks, allowed_texts(el, msgs, provenance))


def attribute(run, provenance, fam):
    """Earliest agent element not supported by what it may draw on.
    Fallback: first agent element containing the final answer; else last agent."""
    msgs = run["messages"]
    for a in range(1, len(msgs)):
        for e in msgs[a]["elements"]:
            if not supported(e, msgs, provenance, fam):
                return a, e["id"], "support-check"
    for a in range(1, len(msgs)):
        for e in msgs[a]["elements"]:
            if not e.get("final") and fam.contains_final(e["text"], run["final"]):
                return a, e["id"], "value-search"
    last = msgs[-1]["elements"][-1]["id"] if msgs[-1]["elements"] else f"M{len(msgs)-1}.F"
    return len(msgs) - 1, last, "default-last"


def false_alarm(run, provenance, fam):
    """On a CLEAN, CORRECT run the rule should find nothing unsupported.
    Returns True if it flags some element anyway (a false alarm)."""
    msgs = run["messages"]
    return any(not supported(e, msgs, provenance, fam) for m in msgs[1:] for e in m["elements"])


# ---------------------------------------------------------------- provenance quality
def source_fidelity(run, fam):
    """Coverage = share of agent elements declaring a source. Fidelity = among
    those, share whose needed tokens all appear in the cited sources."""
    msgs = run["messages"]
    els = [e for m in msgs[1:] for e in m["elements"] if not e["id"].endswith(".R")]
    with_src = [e for e in els if e["sources"]]
    faithful = partial = 0
    for e in with_src:
        _, computed = fam.internal(e["text"])
        toks = fam.tokens(e["text"]) - computed
        faithful += fam.covered(toks, allowed_texts(e, msgs, True))
        declared = []
        for s in e["sources"]:
            k = msg_index(s)
            if k < len(msgs):
                declared += [(x["text"], k == 0) for x in msgs[k]["elements"] if x["id"] == s]
        partial += (not toks) or fam.any_covered(toks, declared)
    return {"elements": len(els), "covered": len(with_src), "faithful": faithful, "partial": partial}

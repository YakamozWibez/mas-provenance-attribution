"""Message format: parsing agent outputs into elements, rendering inputs.

Every message m (0 = task input, 1..3 = agents) is a list of elements with id
"M{m}.E{k}". Element text is stored WITHOUT source annotations.

Structured output (one element per line):
    REASONING: ...                      (not an element, not citable)
    E1: <content> | SOURCE: M0.E1, M1.E2
    FINAL: <answer> | SOURCE: M2.E2     (last agent only)
Free output: plain sentences, sources inline as [M0.E1, M1.E2];
the last agent ends with "FINAL: <answer>" (same marker in all conditions).
"""
import re

ID_RE = re.compile(r"M\d+\.E\d+")
INLINE_SRC_RE = re.compile(r"\[\s*(M\d+\.E\d+(?:\s*[,;]\s*M\d+\.E\d+)*)\s*\]")
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


def _ids(s):
    return ID_RE.findall(s or "")


def msg_index(eid):
    return int(eid[1:].split(".")[0])


def elem_index(eid):
    tail = eid.split(".")[1]
    if tail == "R":
        return 0  # reasoning comes first
    return int(tail[1:]) if tail.startswith("E") else 10 ** 6  # final element last


def parse_output(text, structured, m):
    """Return dict(elements=[{id,text,sources}], final=str|None, valid=bool)."""
    elements, final, final_sources = [], None, []
    lines = [l.strip() for l in (text or "").splitlines() if l.strip()]
    body = []
    for l in lines:
        mm = re.match(r"^\**\s*FINAL\s*\**\s*:\s*\**\s*(.*)$", l, flags=re.I)
        if mm:
            rest = mm.group(1)
            if "|" in rest:
                val, src = rest.split("|", 1)
                final, final_sources = val.strip(), _ids(src)
            else:
                final_sources = [i for g in INLINE_SRC_RE.findall(rest) for i in _ids(g)]
                final = INLINE_SRC_RE.sub("", rest).strip()
        else:
            body.append(l)
    reasoning = []
    if structured:
        in_reason = False
        for l in body:
            if re.match(r"^\**\s*REASONING\s*\**\s*:", l, flags=re.I):
                in_reason = True
                reasoning.append(re.sub(r"^\**\s*REASONING\s*\**\s*:\s*\**\s*", "", l, flags=re.I))
                continue
            if re.match(r"^\**\s*E\d+\s*\**\s*[:.)]", l):
                in_reason = False
            elif in_reason:
                reasoning.append(l)
                continue
            mm = re.match(r"^\**\s*E\d+\s*\**\s*[:.)]\s*\**\s*(.*)$", l)
            if not mm:
                continue  # REASONING or noise
            content = mm.group(1)
            sm = re.search(r"\|\s*SOURCES?\s*:", content, flags=re.I)
            if sm:
                sources = _ids(content[sm.end():])
                content = content[:sm.start()]
            else:
                sources = [i for g in INLINE_SRC_RE.findall(content) for i in _ids(g)]
                content = INLINE_SRC_RE.sub("", content)
            elements.append({"text": content.strip(), "sources": sources})
    else:
        joined = " ".join(body)
        for s in SENT_SPLIT.split(joined):
            s = s.strip()
            if not s:
                continue
            sources = [i for g in INLINE_SRC_RE.findall(s) for i in _ids(g)]
            elements.append({"text": INLINE_SRC_RE.sub("", s).strip(), "sources": sources})
    for k, e in enumerate(elements, 1):
        e["id"] = f"M{m}.E{k}"
    if reasoning:
        elements.insert(0, {"id": f"M{m}.R", "text": " ".join(reasoning).strip(), "sources": []})
    if final is not None:
        elements.append({"id": f"M{m}.F", "text": final, "sources": final_sources, "final": True})
    return {"elements": elements, "final": final, "valid": len(elements) > 0}


def message_from_units(units):
    """Task input from a list of text units (sentences)."""
    return {"elements": [{"id": f"M0.E{k}", "text": u, "sources": []} for k, u in enumerate(units, 1)],
            "final": None, "valid": True}


def split_sentences(text):
    return [s.strip() for s in SENT_SPLIT.split(text) if s.strip()]


def render_inputs(messages, header0="TASK", skip_task=False):
    """Previous messages with ids. Declared sources are NOT rendered, so the
    content given to an agent is identical across conditions."""
    out = []
    for m, msg in enumerate(messages):
        if skip_task and m == 0:
            continue
        out.append(f"--- {header0 if m == 0 else f'MESSAGE FROM AGENT {m}'} ---")
        for e in msg["elements"]:
            out.append(f"[{e['id']}] FINAL: {e['text']}" if e.get("final") else f"[{e['id']}] {e['text']}")
    return "\n".join(out)

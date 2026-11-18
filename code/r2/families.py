"""Task families. Each family defines: task input, answer check, which tokens of
an element need support, internal consistency (math), and how to inject errors.

MathFamily: GSM8K-style word problems. Tokens = numbers.
QAFamily: HotpotQA-style multi-document questions. Tokens = named entities/years.
"""
import ast
import random
import re
import string

from .fmt import message_from_units, split_sentences

# ------------------------------------------------------------------ numbers
WORDS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
    "fifteen sixteen seventeen eighteen nineteen twenty".split())}
TENS = {"thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
WORDS.update(TENS)
WORDS.update({"dozen": 12, "hundred": 100, "thousand": 1000})
NUM_TOKEN = re.compile(r"(?<![\w.])-?\$?\d[\d,]*(?:\.\d+)?%?(?![\w])")
LABEL_RE = re.compile(r"\b(?:step|item|part|e|agent|message|equation)\s*#?\d+\b|#\d+|\b\d+(?:st|nd|rd|th)\b", re.I)


def norm_num(s):
    s = s.replace("$", "").replace(",", "").rstrip("%")
    try:
        v = float(s)
    except ValueError:
        return None
    return str(int(v)) if abs(v - round(v)) < 1e-9 else f"{v:.6g}"


def numbers(text, words=False):
    t = LABEL_RE.sub(" ", text or "")
    out = [norm_num(x) for x in NUM_TOKEN.findall(t)]
    out = [x for x in out if x is not None]
    if words:
        low = t.lower()
        for w, v in WORDS.items():
            if re.search(rf"\b{w}\b", low):
                out.append(str(v))
        for tw, tv in TENS.items():
            for u, uv in list(WORDS.items())[1:10]:
                if re.search(rf"\b{tw}[- ]{u}\b", low):
                    out.append(str(tv + uv))
        if re.search(r"\bhalf\b", low):
            out += ["0.5", "1", "2"]
        for w, v in (("halves", 2), ("thirds?", 3), ("quarters?", 4), ("fourths?", 4), ("fifths?", 5),
                     ("sixths?", 6), ("eighths?", 8), ("tenths?", 10)):
            if re.search(rf"\b{w}\b", low):
                out += [str(v), "1"]
        for w, v in (("double", 2), ("doubles", 2), ("doubled", 2), ("doubling", 2), ("twice", 2), ("triple", 3),
                     ("tripled", 3), ("quadruple", 4)):
            if re.search(rf"\b{w}\b", low):
                out.append(str(v))
    return out


def pool_numbers(text):
    """Numbers a text makes available as a source: digits, number words,
    multiplier words, and percentages also as fractions (20% -> 0.2)."""
    t = LABEL_RE.sub(" ", text or "")
    out = set(numbers(t, words=True))
    for m in re.finditer(r"(?<![\w./])(\d+)/(\d+)(?![\w./])", t):
        if float(m.group(2)):
            out.add(norm_num(str(float(m.group(1)) / float(m.group(2)))))
    fr = {"half": 2, "halves": 2, "third": 3, "thirds": 3, "quarter": 4, "quarters": 4, "fourth": 4,
          "fourths": 4, "fifth": 5, "fifths": 5, "tenth": 10, "tenths": 10}
    nu = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4}
    for m in re.finditer(r"\b(a|an|one|two|three|four)[- ](half|halves|thirds?|quarters?|fourths?|fifths?|tenths?)\b", t.lower()):
        out.add(norm_num(str(nu[m.group(1)] / fr[m.group(2)])))
    for m in re.finditer(r"(\d[\d,]*(?:\.\d+)?)\s*(?:%|percent\b|per cent\b)", t, flags=re.I):
        v = norm_num(m.group(1))
        if v is not None:
            out.add(norm_num(str(float(v) / 100)))
    return out


def _close(a, b):
    return abs(a - b) <= 1e-6 * max(1.0, abs(b)) or round(a, 2) == round(b, 2)


RESULT_CUE = re.compile(
    r"(?:\bequals?\b|\bequaling\b|\bis\b|\bare\b|\bgives?\b|\bgiving\b|\bresults? in\b|\bresulting in\b|"
    r"\btotals?\b|\btotaling\b|\byields?\b|\bcomes to\b|\bmakes?\b|\bmaking\b|\bgets?\b|\bleaves?\b|"
    r"\bleaving\b|\bto get\b|\bto find\b|\brounds? to\b|\brounded to\b|\bcomes out to\b|\bfor a total of\b|\byielding\b)\s+(?:[A-Za-z'’-]+\s+){0,4}"
    r"\$?(\d[\d,]*(?:\.\d+)?)", re.I)


def derived(text):
    """Results of verbal arithmetic: a number right after a result word
    ('which equals 25.00', 'resulting in $8.00') that is obtained from numbers
    written BEFORE it in the same text. Operands are never marked, so they
    still need support from the sources."""
    t = LABEL_RE.sub(" ", text or "")
    out = set()
    for m in RESULT_CUE.finditer(t):
        n = norm_num(m.group(1))
        if n is None:
            continue
        before = t[:m.start(1)]
        prev = [float(x) for x in numbers(before)]
        prev += [float(x) for x in numbers(before, words=True)][len(numbers(before)):]
        for pm in re.finditer(r"(\d[\d,]*(?:\.\d+)?)\s*(?:%|percent\b)", before, flags=re.I):
            prev.append(float(norm_num(pm.group(1))) / 100)
        if len(prev) < 1:
            continue
        v = float(n)
        cands = [sum(prev), _prod(prev)] + [round(x) for x in prev] + [round(x, 2) for x in prev]
        if any(_close(c, v) for c in cands[2:]):
            out.add(n)
            continue
        digits = [float(x) for x in numbers(before)]
        if digits:
            cands += [sum(digits), _prod(digits)]
        for a in range(len(prev)):
            for b in range(len(prev)):
                if a != b:
                    x, y = prev[a], prev[b]
                    cands += [x + y, x - y, x * y] + ([x / y] if y else [])
                    if len(prev) <= 4:  # few operands only: keeps chance matches rare
                        for c in range(len(prev)):
                            if c != a and c != b:
                                z = prev[c]
                                cands += [x * y / z if z else x * y, x + y + z, x + y - z, x * y * z]
        if len(prev) >= 2 and any(_close(c, v) for c in cands):
            out.add(n)
    for m in re.finditer(r"round\(\s*\$?([\d,]*\.?\d+)\s*\)\s*=\s*\$?([\d,]*\.?\d+)", t):
        if round(float(m.group(1).replace(",", ""))) == float(m.group(2).replace(",", "")):
            out.add(norm_num(m.group(2)))
    return out


def _prod(xs):
    p = 1.0
    for x in xs:
        p *= x
    return p


class _Eval(ast.NodeVisitor):
    def visit(self, node):
        if isinstance(node, ast.Expression):
            return self.visit(node.body)
        if isinstance(node, ast.BinOp) and type(node.op) in (ast.Add, ast.Sub, ast.Mult, ast.Div):
            a, b = self.visit(node.left), self.visit(node.right)
            return {ast.Add: a + b, ast.Sub: a - b, ast.Mult: a * b,
                    ast.Div: a / b if b else float("nan")}[type(node.op)]
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            a, b = self.visit(node.left), self.visit(node.right)
            if abs(b) > 12:
                raise ValueError("exponent too large")
            return a ** b
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -self.visit(node.operand)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError("bad node")


EXPR_CHARS = set("0123456789.,$ ()+-*/×÷x%")


def _clean_expr(p):
    s = p.replace(",", "").replace("$", "").replace("×", "*").replace("÷", "/").replace("^", "**")
    s = re.sub(r"(?<=\d)\s*[xX]\s*(?=\(?\$?\d)", " * ", s)
    s = re.sub(r"(\d+(?:\.\d+)?)\s*%", r"(\1/100)", s)
    s = re.sub(r"/\s*[A-Za-z]+", " ", s)          # per-unit: "/pack", "/day"
    s = re.sub(r"[A-Za-z_'’:?!;\"]+", " ", s)       # words and punctuation
    s = re.sub(r"\.(?!\d)", " ", s)                 # full stops not inside a number
    return re.sub(r"\s+", " ", s).strip()


def _try(s):
    try:
        return _Eval().visit(ast.parse(s, mode="eval"))
    except Exception:
        return None


def _tail(p):
    """Value of the longest expression at the END of a text piece (left of '=')."""
    toks = _clean_expr(p).split(" ")
    for k in range(len(toks)):
        cand = " ".join(toks[k:]).strip()
        if cand and re.search(r"\d", cand):
            v = _try(cand)
            if v is not None:
                return v, bool(re.search(r"\d\)?\s*[-+*/]\s*\(?\d", cand)), cand
    return None, False, None


def _lead(p):
    """Value of the longest expression at the START of a text piece (right of '=')."""
    toks = _clean_expr(p).split(" ")
    for k in range(len(toks), 0, -1):
        cand = " ".join(toks[:k]).strip()
        if cand and re.search(r"\d", cand):
            v = _try(cand)
            if v is not None:
                single = re.fullmatch(r"-?\d+(?:\.\d+)?", cand) is not None
                return v, single, cand
    return None, False, None


def arithmetic(text):
    """Check written computations 'expression = value', including units and chains.
    Return (consistent, computed_values). Operands are never marked as computed."""
    t = LABEL_RE.sub(" ", text or "")
    t = re.sub(r"round\(([^)]*)\)", r"\1", t)
    computed, consistent = set(), True
    parts = t.split("=")
    algebra = re.compile(r"\b\d+\s*(?:[a-wyzA-WYZ]|[xX](?!\s*\(?\$?\d))\b")
    num = re.compile(r"\d+(?:\.\d+)?")
    for k in range(1, len(parts)):
        if algebra.search(parts[k - 1]) or algebra.search(parts[k].split(".")[0]):
            continue  # equations with unknowns (7x + 12 = 341) are not checked
        left, has_op, lcand = _tail(parts[k - 1])
        if left is None or not has_op:
            continue
        right, single, cand = _lead(parts[k])
        if right is None:
            continue
        if _close(left, right) or (round(left) == right and "round" in text.lower()):
            if single:
                computed.add(norm_num(cand))
            else:  # chain: intermediate results written in the next expression
                before = {norm_num(x) for x in num.findall(lcand or "")}
                computed |= {norm_num(x) for x in num.findall(cand)} - before
        else:
            consistent = False
    return consistent, computed | derived(text)


# ------------------------------------------------------------------ math family
class MathFamily:
    name = "math"
    header0 = "TASK"
    ROLES = {
        1: ("planner", "Read the task. List every quantity given in the task, one per item, writing each "
                       "value in digits, and state which operations are needed. Do not compute the final answer."),
        2: ("calculator", "Using the planner's items, compute the solution step by step. Write every "
                          "computation as an explicit expression with its result, for example 16 - 3 - 4 = 9."),
        3: ("checker", "Check the previous agents' work and give the final numeric answer."),
    }
    FINAL_HINT = "<number>"

    def task_message(self, task):
        return message_from_units(split_sentences(task["question"]))

    def answer_ok(self, final, task):
        nums = numbers(final or "")
        return bool(nums) and nums[-1] == norm_num(str(task["answer"]))

    def tokens(self, text):
        return set(numbers(text))

    def pool_tokens(self, text, is_task):
        return set(numbers(text, words=is_task))

    def internal(self, text):
        return arithmetic(text)

    def replaceable(self, text):
        return [x for x in NUM_TOKEN.findall(LABEL_RE.sub(" ", text)) if norm_num(x) is not None]

    def fabricate(self, target, used, rng):
        base = float(norm_num(target))
        for _ in range(1000):
            v = int(round(base)) + rng.choice([-1, 1]) * rng.randint(7, 97)
            if v > 0 and str(v) not in used:
                return str(v)
        return None

    def misbind_pool(self, task):
        return sorted(set(numbers(task["question"])), key=lambda x: float(x))

    def used_index(self, task, msgs):
        used = set(numbers(task["question"], words=True))
        for m in msgs:
            for e in m["elements"]:
                used.update(numbers(e["text"]))
        return used

    def covered(self, toks, allowed):
        pool = set()
        for t, is_task in allowed:
            pool |= pool_numbers(t)
        return all(t in pool for t in toks)

    def any_covered(self, toks, allowed):
        pool = set()
        for t, is_task in allowed:
            pool |= pool_numbers(t)
        return any(t in pool for t in toks)

    def contains_final(self, text, final):
        f = numbers(final or "")
        return bool(f) and f[-1] in numbers(text)

    def same(self, a, b):
        return norm_num(a) == norm_num(b)


# ------------------------------------------------------------------ QA family
STOP = set("""The A An He She It They This That These Those In On At His Her Their Its There Which Who
What When Where Why How Yes No Both So Therefore Thus Final Answer Step First Second Third Then Also
However And But Or If As For From By With To Of Fact Facts Document Documents Question Agent Message
Based According Since Because While After Before During Each Every Not Answer Reasoning Source
Sources Item Items One Two""".split())
ENT_RE = re.compile(r"\b[A-Z][\w'’\-]*(?:\s+(?:of|the|de|del|da|van|von|der|and|&)?\s*[A-Z][\w'’\-]*)*|\b\d{3,4}\b")


def _norm(s):
    s = s.lower().replace("’", "'")
    s = "".join(ch for ch in s if ch not in set(string.punctuation) - {"'"})
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def entities(text):
    out = []
    for m in ENT_RE.finditer(text or ""):
        e = m.group(0).strip()
        words = e.split()
        while words and words[0] in STOP:
            words = words[1:]
        e = " ".join(words)
        if e and e not in STOP and len(e) > 1:
            out.append(e)
    return out


class QAFamily:
    name = "qa"
    header0 = "DOCUMENTS AND QUESTION"
    ROLES = {
        1: ("reader", "Read the documents and the question. Extract every fact needed to answer, one per "
                      "item, copying names exactly as written in the documents. Do not answer the question."),
        2: ("linker", "Using the reader's facts, connect them step by step to reach the answer. Each step "
                      "must say what follows from the facts it uses."),
        3: ("answerer", "Check the previous agents' steps and give the final answer as a short phrase."),
    }
    FINAL_HINT = "<short answer>"

    def task_message(self, task):
        units = []
        for title, sents in task["context"]:
            for s in sents:
                s = s.strip()
                if s:
                    units.append(f"({title}) {s}")
        units.append(f"QUESTION: {task['question']}")
        return message_from_units(units)

    def answer_ok(self, final, task):
        p, g = _norm(final or ""), _norm(task["answer"])
        if not p or not g:
            return False
        return p == g or (re.search(rf"\b{re.escape(g)}\b", p) is not None and len(p.split()) <= len(g.split()) + 3)

    def tokens(self, text):
        return {_norm(e) for e in entities(text)} - {""}

    def internal(self, text):
        return True, set()

    def replaceable(self, text):
        return entities(text)

    def fabricate(self, target, used, rng):
        syl = ["vel", "mor", "kas", "trin", "ald", "oro", "bri", "sen", "tal", "qui", "dor", "mex", "ula"]
        for _ in range(1000):
            name = " ".join(("".join(rng.choice(syl) for _ in range(2))).capitalize()
                            for _ in range(max(1, len(target.split()))))
            if _norm(name) not in used:
                return name  # used is a normalised text blob: substring test
        return None

    def misbind_pool(self, task):
        ents = []
        for title, sents in task["context"]:
            ents += entities(title) + [e for s in sents for e in entities(s)]
        seen, out = set(), []
        for e in ents:
            if _norm(e) not in seen:
                seen.add(_norm(e)); out.append(e)
        return out

    def used_index(self, task, msgs):
        blob = [self.task_message(task)["elements"][i]["text"] for i in range(len(self.task_message(task)["elements"]))]
        for m in msgs:
            blob += [e["text"] for e in m["elements"]]
        return _norm(" ".join(blob))

    def covered(self, toks, allowed):
        texts = [_norm(t) for t, _ in allowed]
        return all(any(re.search(rf"\b{re.escape(tok)}\b", p) for p in texts) for tok in toks)

    def any_covered(self, toks, allowed):
        texts = [_norm(t) for t, _ in allowed]
        return any(any(re.search(rf"\b{re.escape(tok)}\b", p) for p in texts) for tok in toks)

    def contains_final(self, text, final):
        f = _norm(final or "")
        return bool(f) and re.search(rf"\b{re.escape(f)}\b", _norm(text)) is not None

    def same(self, a, b):
        return _norm(a) == _norm(b)


FAMILIES = {"math": MathFamily, "qa": QAFamily}

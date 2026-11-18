"""LLM interfaces.

MockMath / MockQA simulate the three agents with KNOWN error rates, so the
pipeline's measurements can be checked against ground truth. They validate the
code, not the research hypothesis.

RealLLM wraps OpenAI-compatible and Anthropic APIs. NOT TESTED from this
workspace (no network to the APIs). Keys are read only from environment
variables: OPENAI_API_KEY (optional OPENAI_BASE_URL for compatible providers)
or ANTHROPIC_API_KEY.
"""
import os
import random
import re
import time

NUM = re.compile(r"(?<![\w.])-?\d+(?![\w])")


class _MockBase:
    def __init__(self, seed=0, p_omit_source=0.05, p_wrong_source=0.10, p_invalid=0.01):
        self.rng = random.Random(seed)
        self.p_omit, self.p_wrong, self.p_invalid = p_omit_source, p_wrong_source, p_invalid

    @staticmethod
    def _cond(system):
        m = re.search(r"structured=(\d) provenance=(\d)", system)
        return bool(int(m.group(1))), bool(int(m.group(2)))

    @staticmethod
    def _role(system):
        return int(re.search(r"You are agent (\d)", system).group(1))

    @staticmethod
    def _items(user, m):
        return re.findall(rf"\[(M{m}\.E\d+)\] (.*)", user)

    def _src(self, true_ids, all_ids):
        if self.rng.random() < self.p_omit:
            return []
        ids = list(true_ids)
        if ids and self.rng.random() < self.p_wrong:
            others = [x for x in all_ids if x not in ids]
            if others:
                ids[self.rng.randrange(len(ids))] = self.rng.choice(others)
        return ids

    def _emit(self, items, structured, provenance, final=None):
        lines = []
        if structured:
            lines.append("REASONING: working through the items in order.")
            for k, (txt, src) in enumerate(items, 1):
                s = f" | SOURCE: {', '.join(src)}" if provenance and src else ""
                lines.append(f"E{k}: {txt}{s}")
        else:
            lines.append(" ".join(f"{txt}" + (f" [{', '.join(src)}]" if provenance and src else "") + "."
                                  for txt, src in items))
        if final is not None:
            val, src = final
            tail = ""
            if provenance and src:
                tail = f" | SOURCE: {', '.join(src)}" if structured else f" [{', '.join(src)}]"
            lines.append(f"FINAL: {val}{tail}")
        return "\n".join(lines)

    @staticmethod
    def _usage(system, user, text):
        return {"in": (len(system) + len(user)) // 4, "out": len(text) // 4}

    def call(self, system, user, max_tokens=600):
        structured, provenance = self._cond(system)
        if self.rng.random() < self.p_invalid:
            text = "I cannot help with that."
        else:
            text = self._agent(self._role(system), user, structured, provenance,
                               re.findall(r"\[(M\d+\.E\d+)\]", user))
        return text, self._usage(system, user, text)


class MockMath(_MockBase):
    OPS = {"buys": "+", "finds": "+", "receives": "+", "gives away": "-", "loses": "-", "eats": "-"}

    def __init__(self, p_calc_error=0.05, struct_calc_penalty=0.0, **kw):
        super().__init__(**kw)
        self.p_calc, self.struct_pen = p_calc_error, struct_calc_penalty

    def _agent(self, role, user, structured, provenance, all_ids):
        if role == 1:
            items = []
            for i, t in self._items(user, 0):
                for n in NUM.findall(t):
                    op = next((o for w, o in self.OPS.items() if w in t), None)
                    label = "starting amount" if op is None else f"amount combined with operation {op}"
                    items.append((f"The {label} is {n}", self._src([i], all_ids)))
            return self._emit(items, structured, provenance)
        if role == 2:
            vals = []
            for i, t in self._items(user, 1):
                nums = NUM.findall(t)
                if nums:
                    op = re.search(r"operation ([+\-])", t)
                    vals.append((i, int(nums[-1]), op.group(1) if op else None))
            p_err = self.p_calc + (self.struct_pen if structured else 0)
            items = []
            if vals:
                cur_id, cur, _ = vals[0]
                for k, (i, v, op) in enumerate(vals[1:], 1):
                    res = cur + v if (op or "+") == "+" else cur - v
                    if self.rng.random() < p_err:
                        res += self.rng.choice([-3, -2, -1, 1, 2, 3])
                    items.append((f"Then {cur} {op or '+'} {v} = {res}", self._src([cur_id, i], all_ids)))
                    cur, cur_id = res, f"M2.E{k}"
            return self._emit(items, structured, provenance)
        steps = [(i, t) for i, t in self._items(user, 2) if NUM.findall(t)]
        if not steps:
            return self._emit([("No steps were provided", [])], structured, provenance, final=("0", []))
        i, t = steps[-1]
        val = NUM.findall(t)[-1]
        return self._emit([(f"The last step gives {val}", self._src([i], all_ids))], structured, provenance,
                          final=(val, self._src([i], all_ids)))


class MockQA(_MockBase):
    """Two-hop questions: '(A) X was born in Y.' '(B) Y is a town in Z.' Q: region of X's birthplace?"""

    def __init__(self, p_link_error=0.05, struct_link_penalty=0.0, **kw):
        super().__init__(**kw)
        self.p_link, self.struct_pen = p_link_error, struct_link_penalty

    def _agent(self, role, user, structured, provenance, all_ids):
        if role == 1:
            q = re.search(r"QUESTION: In which region was (.+?) born\?", user)
            person = q.group(1) if q else ""
            docs = self._items(user, 0)
            f1 = next(((i, t) for i, t in docs if f"{person} was born in" in t), None)
            items = []
            if f1:
                place = re.search(r"was born in ([A-Z][\w]+)", f1[1]).group(1)
                items.append((f"{person} was born in {place}", self._src([f1[0]], all_ids)))
                f2 = next(((i, t) for i, t in docs if t.split(") ", 1)[-1].startswith(f"{place} is a town")), None)
                if f2:
                    region = re.search(r"in the region of ([A-Z][\w]+)", f2[1]).group(1)
                    items.append((f"{place} is a town in the region of {region}", self._src([f2[0]], all_ids)))
            return self._emit(items, structured, provenance)
        if role == 2:
            facts = self._items(user, 1)
            if len(facts) < 2:
                return self._emit([("The facts are insufficient", [])], structured, provenance)
            (i1, t1), (i2, t2) = facts[0], facts[1]
            m1 = re.search(r"^(.+?) was born in ([A-Z][\w]+)", t1)
            m2 = re.search(r"region of ([A-Z][\w]+)", t2)
            if not (m1 and m2):
                return self._emit([("The facts cannot be linked", [])], structured, provenance)
            region = m2.group(1)
            p_err = self.p_link + (self.struct_pen if structured else 0)
            if self.rng.random() < p_err:
                region = "Nowhere"
            return self._emit([(f"{m1.group(1)} was born in {m1.group(2)}, which lies in the region of {region}",
                                self._src([i1, i2], all_ids))], structured, provenance)
        steps = self._items(user, 2)
        if not steps:
            return self._emit([("No answer", [])], structured, provenance, final=("unknown", []))
        i, t = steps[-1]
        m = re.search(r"region of ([A-Z][\w]+)", t)
        region = m.group(1) if m else "unknown"
        return self._emit([(f"The answer is the region of {region}", self._src([i], all_ids))],
                          structured, provenance, final=(region, self._src([i], all_ids)))


MockLLM = MockMath  # backwards compatibility


class RealLLM:
    """Untested here. provider in {'openai','anthropic'}; key from env only."""

    def __init__(self, provider, model, temperature=0.0, max_retries=5, extra_body=None):
        self.provider, self.model, self.temperature, self.max_retries = provider, model, temperature, max_retries
        self.extra_body = extra_body or {}
        if provider == "openai":
            from openai import OpenAI  # pip install openai
            kw = {"api_key": os.environ["OPENAI_API_KEY"]}
            if os.environ.get("OPENAI_BASE_URL"):
                kw["base_url"] = os.environ["OPENAI_BASE_URL"]
            self.client = OpenAI(**kw)
        elif provider == "anthropic":
            import anthropic  # pip install anthropic
            self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        else:
            raise ValueError(provider)

    def _once(self, system, user, max_tokens):
        # The CONDITION line exists only for the mock models; real models never see it.
        system = "\n".join(l for l in system.splitlines() if not l.startswith("CONDITION:"))
        if self.provider == "openai":
            r = self.client.chat.completions.create(
                model=self.model, temperature=self.temperature, max_tokens=max_tokens,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                extra_body=self.extra_body)
            return r.choices[0].message.content or "", {"in": r.usage.prompt_tokens, "out": r.usage.completion_tokens}
        r = self.client.messages.create(model=self.model, system=system, max_tokens=max_tokens,
                                        temperature=self.temperature,
                                        messages=[{"role": "user", "content": user}])
        text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
        return text, {"in": r.usage.input_tokens, "out": r.usage.output_tokens}

    def call(self, system, user, max_tokens=600):
        for k in range(self.max_retries):
            try:
                return self._once(system, user, max_tokens)
            except Exception as e:  # rate limits, transient errors
                code = getattr(e, "status_code", None)
                if code and 400 <= code < 500 and code != 429:
                    raise  # not transient: stop at once
                if k == self.max_retries - 1:
                    raise
                time.sleep(min(60, 2 ** k * 2))

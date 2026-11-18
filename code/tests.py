"""Unit tests on hand-built traces with known answers. Run: python3 tests.py"""
import random
import sys

from r2.families import MathFamily, QAFamily, arithmetic, derived, entities, numbers, pool_numbers
from r2.fmt import parse_output
from r2.llm import MockMath, MockQA
from r2.pipeline import attribute, correct, false_alarm, inject, run_chain, source_fidelity, supported
from r2.tasks import synth_math, synth_qa

fails = 0


def check(name, cond):
    global fails
    print(("OK   " if cond else "FAIL ") + name)
    fails += 0 if cond else 1


M, Q = MathFamily(), QAFamily()

# ---- parsing
s = parse_output("REASONING: x 5\nE1: The amount is 7 | SOURCE: M0.E2\nE2: Then 3 + 7 = 10 | SOURCE: M1.E1, M0.E1\nFINAL: 10 | SOURCE: M2.E2", True, 2)
check("strutturato: ragionamento + 2 elementi + finale", len(s["elements"]) == 4 and s["elements"][0]["id"] == "M2.R" and s["final"] == "10")
check("strutturato: fonti lette", s["elements"][2]["sources"] == ["M1.E1", "M0.E1"])
s2 = parse_output("**E1:** The amount is 7 | Source: M0.E2\n**FINAL:** 7", True, 1)
check("strutturato: varianti con grassetto e maiuscole", s2["elements"][0]["sources"] == ["M0.E2"] and s2["final"] == "7")
f = parse_output("The amount is 7 [M0.E2]. Then 3 + 7 = 10 [M1.E1, M0.E1].\nFINAL: 10 [M2.E2]", False, 2)
check("libero: 2 frasi + finale", len(f["elements"]) == 3 and f["final"] == "10")
check("libero: id tolti dal testo", "M0" not in f["elements"][0]["text"])

# ---- numbers and arithmetic
check("numeri: virgole, dollari, decimali", numbers("She pays $1,200 and 2.50 each") == ["1200", "2.5"])
check("numeri: etichette ignorate", numbers("Step 2: 16 - 3 = 13") == ["16", "3", "13"])
check("numeri: parole nel testo del compito", "3" in numbers("She eats three for breakfast", words=True))
check("aritmetica: catena corretta", arithmetic("16 - 3 - 4 = 9") == (True, {"9"}))
check("aritmetica: errore rilevato", arithmetic("16 - 3 - 4 = 10")[0] is False)
check("aritmetica: $ e moltiplicazione", arithmetic("9 * $2 = $18") == (True, {"18"}))
check("aritmetica: percentuale", arithmetic("20% of nothing; 50 * 20% = 10")[1] == {"10"})

check("aritmetica a parole: risultato riconosciuto", arithmetic("multiplying 5 by $5.00, which equals $25.00")[1] == {"25"})
check("aritmetica a parole: operandi NON marcati", "50" not in arithmetic("Then 12 + 50 = 62")[1] and "50" not in derived("adding 12 and 50 gives 62"))
check("fonti: 20% vale anche 0.2", "0.2" in pool_numbers("everything was 20% off"))
check("fonti: 'double' vale 2", "2" in pool_numbers("He likes to double that amount."))

# ---- math attribution: misbinding vs fabricated
task = {"question": "Anna has 12 apples. Then Anna buys 7 apples. Then Anna loses 3 apples. How many?", "answer": 16}
m0 = M.task_message(task)
m1 = parse_output("E1: The starting amount is 12 | SOURCE: M0.E1\nE2: She buys 3 | SOURCE: M0.E2\nE3: She loses 3 | SOURCE: M0.E3", True, 1)
m2 = parse_output("E1: Then 12 + 3 = 15 | SOURCE: M1.E1, M1.E2\nE2: Then 15 - 3 = 12 | SOURCE: M2.E1, M1.E3", True, 2)
m3 = parse_output("E1: The last step gives 12 | SOURCE: M2.E2\nFINAL: 12 | SOURCE: M2.E2", True, 3)
run = {"messages": [m0, m1, m2, m3], "final": "12"}
check("matematica: scambio trovato CON fonti", attribute(run, True, M)[1] == "M1.E2")
check("matematica: scambio NON trovato senza fonti", attribute(run, False, M)[1] != "M1.E2")
m1b = parse_output("E1: The starting amount is 12 | SOURCE: M0.E1\nE2: She buys 50 | SOURCE: M0.E2\nE3: She loses 3 | SOURCE: M0.E3", True, 1)
m2b = parse_output("E1: Then 12 + 50 = 62 | SOURCE: M1.E1, M1.E2\nE2: Then 62 - 3 = 59 | SOURCE: M2.E1, M1.E3", True, 2)
runb = {"messages": [m0, m1b, m2b, parse_output("E1: The last step gives 59 | SOURCE: M2.E2\nFINAL: 59 | SOURCE: M2.E2", True, 3)], "final": "59"}
check("matematica: inventato trovato con fonti", attribute(runb, True, M)[1] == "M1.E2")
check("matematica: inventato trovato senza fonti", attribute(runb, False, M)[1] == "M1.E2")
m1c = parse_output("E1: The starting amount is 12 | SOURCE: M0.E3\nE2: She buys 3 | SOURCE: M0.E2\nE3: She loses 3 | SOURCE: M0.E3", True, 1)
runc = {"messages": [m0, m1c, m2, m3], "final": "12"}
check("matematica: fonte falsa porta ad attribuzione sbagliata", attribute(runc, True, M)[1] == "M1.E1")
fq = source_fidelity(runc, M)
check("matematica: fedelta conta la fonte falsa", fq["faithful"] < fq["covered"])

# ---- QA family
qt = {"question": "In which region was Velmor born?", "answer": "Kastrin",
      "context": [("Velmor", ["Velmor was born in Oroald."]), ("Oroald", ["Oroald is a town in the region of Kastrin."]),
                  ("Senbri", ["Senbri was born in Talqui."]), ("Talqui", ["Talqui is a town in the region of Dorpax."])]}
q0 = Q.task_message(qt)
check("QA: entita estratte", set(entities("Velmor was born in Oroald, in the region of Kastrin")) >= {"Velmor", "Oroald", "Kastrin"})
check("QA: risposta normalizzata", Q.answer_ok("the Kastrin", qt) and not Q.answer_ok("Dorpax", qt))
ids = {e["text"]: e["id"] for e in q0["elements"]}
q1 = parse_output("E1: Velmor was born in Talqui | SOURCE: M0.E1\nE2: Oroald is a town in the region of Kastrin | SOURCE: M0.E2", True, 1)
run_q = {"messages": [q0, q1], "final": "Dorpax"}
check("QA: scambio di entita trovato CON fonti", attribute(run_q, True, Q)[1] == "M1.E1")
check("QA: scambio di entita NON trovato senza fonti", attribute(run_q, False, Q)[1] != "M1.E1")

# ---- end-to-end with perfect mocks
for fam, Mock, gen, err in ((M, MockMath, synth_math, "p_calc_error"), (Q, MockQA, synth_qa, "p_link_error")):
    llm = Mock(seed=1, p_omit_source=0, p_wrong_source=0, p_invalid=0, **{err: 0})
    ts = gen(15, seed=2)
    ok = all(correct(run_chain(t, s_, p_, llm, fam), t, fam) for t in ts for s_ in (0, 1) for p_ in (0, 1))
    check(f"{fam.name}: modello finto perfetto risolve tutto", ok)
    fa = [false_alarm(run_chain(t, s_, p_, llm, fam), p_, fam) for t in ts for s_ in (0, 1) for p_ in (0, 1)]
    check(f"{fam.name}: nessun falso allarme su esecuzioni pulite ({sum(fa)})", not any(fa))
    rng, hits = random.Random(0), []
    for t in ts:
        r = run_chain(t, True, True, llm, fam)
        rec, _ = inject(r, t, True, True, llm, fam, rng, kind="M")
        if rec:
            hits.append(attribute(rec["run"], True, fam)[1] == rec["element"])
    check(f"{fam.name}: scambio attribuito con fonti fedeli ({sum(hits)}/{len(hits)})", bool(hits) and all(hits))

print("\nfallimenti:", fails)
sys.exit(1 if fails else 0)

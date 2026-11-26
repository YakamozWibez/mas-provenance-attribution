"""Hand-made cases for ambiguity_star.py (pre-registration 04). Run: python test_ambiguity_star.py"""
from ambiguity_star import align, k_star, ordered_numbers


def el(i, text, sources=()):
    return {"id": i, "text": text, "sources": list(sources)}


def run(task, a1, a2):
    return {"messages": [{"elements": [el(f"M0.E{i+1}", t) for i, t in enumerate(task)]},
                         {"elements": [el(f"M1.E{i+1}", t, s) for i, (t, s) in enumerate(a1)]},
                         {"elements": [el(f"M2.E{i+1}", t, s) for i, (t, s) in enumerate(a2)]}]}


fails = 0


def check(name, got, want):
    global fails
    ok = got == want
    fails += not ok
    print(("ok  " if ok else "FAIL"), name, got, "" if ok else f"(want {want})")


# 1. Dora: rope 6 -> 16 (swap with mother's 16). Twins in agent 1; order clears the 5th.
dora_task = ["A rope costs $6, a game $11 and a ball $2.", "She saved $2 and her mother gave her $16.",
             "How much more does she need?"]
dora_a1 = [("Cost of the rope = $16", ["M0.E1"]), ("Cost of the game = $11", ["M0.E1"]),
           ("Cost of the ball = $2", ["M0.E1"]), ("Savings = $2", ["M0.E2"]), ("Gift = $16", ["M0.E2"])]
dora_a2 = [("Total cost = 16 + 11 + 2 = 29", ["M1.E1", "M1.E2", "M1.E3"]),
           ("Money = 2 + 16 = 18", ["M1.E4", "M1.E5"]), ("Missing = 29 - 18 = 11", ["M2.E1", "M2.E2"])]
r = run(dora_task, dora_a1, dora_a2)
check("dora no sources", k_star(r, "M1.E1", False),
      {"k": 2, "k_starA": 2, "k_starB": 1, "culprit_cleared_A": False, "culprit_cleared_B": False})
check("dora with sources", k_star(r, "M1.E1", True)["k"], 1)

# 2. Apples: 5, 12, 5 -> agent writes 5, 5, 5 (bought 12 -> 5). Three twins; order leaves one.
r = run(["Luca has 5 apples.", "He buys 12 apples.", "He gives away 5 apples."],
        [("5", []), ("5", []), ("5", [])], [("5 + 5 - 5 = 5", [])])
check("apples", k_star(r, "M1.E2", False),
      {"k": 3, "k_starA": 3, "k_starB": 1, "culprit_cleared_A": False, "culprit_cleared_B": False})

# 3. Number words: four, two, 1 -> agent writes 4, 1, 1 (two -> 1). Diagonal alignment.
r = run(["Brandon's phone is four times as old as Ben's.", "Ben's is two times older than Suzy's.",
         "Suzy's phone is 1 year old."], [("4 (multiplier)", []), ("1 (multiplier)", []), ("1 (age)", [])],
        [("Ben = 1 * 1 = 1", []), ("Brandon = 4 * 1 = 4", [])])
check("words", k_star(r, "M1.E2", False),
      {"k": 2, "k_starA": 2, "k_starB": 1, "culprit_cleared_A": False, "culprit_cleared_B": False})

# 4. Error in agent 2 (result changed): its twin is a correct computation, cleared by arithmetic.
r = run(["Tom has 3 boxes of 4 pens.", "He also has 7 pens."], [("3", []), ("4", []), ("7", [])],
        [("3 * 4 = 19", []), ("3 * 4 = 19", [])])
res = k_star(r, "M2.E1", False)
check("agent2 twins both inconsistent", (res["k"], res["k_starA"]), (2, 2))

# 5. Reordered agent: order cue misleads and clears the culprit.
r = run(["A costs 6.", "B costs 16."], [("16", []), ("16", [])], [("16 + 16 = 32", [])])
res = k_star(r, "M1.E2", False)  # suppose the agent listed B first; the culprit is the 2nd
check("reordered: culprit cleared flag", res["culprit_cleared_B"], True)

check("ordered_numbers", ordered_numbers("twenty-five apples, twice as many, $1,200.50"), ["25", "2", "1200.5"])
check("align", align(["6", "11", "2", "2", "16"], ["16", "11", "2", "2", "16"]),
      [False, True, True, True, True])
print("fallimenti:", fails)

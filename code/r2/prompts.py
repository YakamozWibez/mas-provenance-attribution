"""System prompts. Four conditions = structure x provenance. Role text and the
rendered inputs are identical across conditions; only output instructions differ."""
import hashlib

STRUCT = {
    False: "Write in plain sentences. Do not use numbered items, lists or fields.",
    True: ("Use exactly this format, one item per line, and nothing else:\n"
           "REASONING: <your reasoning, written before any item>\n"
           "E1: <item>\nE2: <item>\n...\nEach item must be understandable on its own: say what each value is."),
}
PROV = ("Every item must declare where its information comes from, using the ids shown in brackets "
        "in the input (for example M0.E1 or M1.E2). Cite only ids that contain the information you use. ")
PROV_SYNTAX = {False: "Put the ids in brackets at the end of each sentence, for example [M0.E1, M1.E2].",
               True: "Append to each item: | SOURCE: <ids separated by commas>."}


def system_prompt(fam, role, structured, provenance):
    name, task = fam.ROLES[role]
    parts = [f"ROLE: {name}. You are agent {role} of 3 in a chain. {task}", STRUCT[structured]]
    if provenance:
        parts.append(PROV + PROV_SYNTAX[structured])
    if role == 3:
        fin = f"End with a last line: FINAL: {fam.FINAL_HINT}"
        if provenance:
            fin += " | SOURCE: <ids>" if structured else " [<ids>]"
        parts.append(fin)
    parts.append(f"CONDITION: structured={int(structured)} provenance={int(provenance)}")
    return "\n".join(parts)


def prompts_hash(fam):
    blob = "\n".join(system_prompt(fam, r, s, p) for r in (1, 2, 3) for s in (0, 1) for p in (0, 1))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]

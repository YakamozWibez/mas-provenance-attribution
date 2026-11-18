"""Re-apply the CURRENT attribution rule to an existing results file, without new
model calls. Usage: python rescore.py results/<file>.jsonl [--family math]
Writes <file>_rescored.jsonl and <file>_rescored_report.txt next to it."""
import argparse, json, sys

from r2.experiment import report, summarize
from r2.families import FAMILIES
from r2.pipeline import attribute, false_alarm, source_fidelity

ap = argparse.ArgumentParser()
ap.add_argument("path"); ap.add_argument("--family", default="math"); ap.add_argument("--config", default="config.json")
a = ap.parse_args()
fam = FAMILIES[a.family](); cfg = json.load(open(a.config))
rows = [json.loads(l) for l in open(a.path, encoding="utf8")]
for r in rows:
    run = r["clean"]
    if r["p"]:
        r["fid"] = source_fidelity(run, fam)
    r["false_alarm"] = false_alarm(run, r["p"], fam) if r["ok"][0] else None
    for k, i in r["inj"].items():
        if i.get("decisive"):
            pa, pe, how = attribute(i["run"], r["p"], fam)
            i.update({"pred_agent": pa, "pred_element": pe, "how": how,
                      "hit_agent": pa == i["agent"], "hit_element": pe == i["element"]})
out = a.path.replace(".jsonl", "_rescored.jsonl")
with open(out, "w", encoding="utf8") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
txt = report(summarize(rows, cfg), rows)
open(out.replace(".jsonl", "_report.txt"), "w").write(txt)
print(txt)

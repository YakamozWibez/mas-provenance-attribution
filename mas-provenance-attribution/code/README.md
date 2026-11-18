# Pipeline (v6)

A chain of three agents, four conditions (structure × declared sources) and two task families:
- `math`: multi-step problems (GSM8K test set, 1,319 problems, MIT licence; file in `data/`);
- `qa`: multi-document questions (HotpotQA dev distractor, CC BY-SA 4.0). This file has to be downloaded; see below.

For every task and condition:
1. clean runs, 3 repetitions with majority vote;
2. two kinds of injected error, kept only when they change the final answer:
   - **U**: a fabricated value that appears nowhere;
   - **M**: a misbinding, i.e. a true value taken from another sentence of the task.

Visibility is `"spec"`: only agent 1 sees the task text, so later agents cannot silently fix an error by re-reading it.

## Attribution

- **Deterministic rule** (`r2/pipeline.py`, no LLM). It returns the first element that is not supported by its allowed sources:
  - in the conditions with sources, the allowed sources are the declared sources plus the agent's own earlier items;
  - otherwise, everything that comes earlier.
  Support for numbers is checked through explicit arithmetic, verbal derivations and numbers taken from the task.
- **LLM judge** (`judge.py`). A standard post-hoc attributor ("all-at-once", as in Who&When), run on the logs already collected. It reads the task, the messages with their element ids, any declared sources, the wrong answer and the correct answer.

## Commands

```
python tests.py                                                     # 32 tests, mock models, free
python run_pilot.py --family math --data synth --llm mock           # free code check
python run_pilot.py --family math --data gsm8k:data/gsm8k_test.jsonl --llm openai --model MODEL --dry-run
python run_pilot.py --family math --data gsm8k:data/gsm8k_test.jsonl --llm openai --model MODEL
python judge.py --results results/<file>.jsonl --data gsm8k:data/gsm8k_test.jsonl --model MODEL \
                --price-in X --price-out Y [--max-tokens 4000] [--dry-run]
python rescore.py results/<file>.jsonl                              # re-apply the current rule, no new calls
```

For OpenAI-compatible providers (Together), set `export OPENAI_BASE_URL=https://api.together.xyz/v1` and enter the key with `read -s OPENAI_API_KEY && export OPENAI_API_KEY`. Keys never go into files.

## Outputs (`results/`)

| File | Content |
|---|---|
| `<tag>.jsonl` | one row per task and condition, with all raw outputs; a run resumes where it stopped |
| `<tag>_report.txt` | per-condition table and gate |
| `<tag>_manual_check_20.txt` | 20 cases to check by hand |
| `<tag>_info.json` | date, model, hashes of prompts and config, full config |
| `<tag>_judge_<model>.jsonl` and `_report.txt` | judge decisions and summary |

The report labels are in Italian:
- `libera`: free condition;
- `strutt`: structured condition;
- `+prov`: with declared sources;
- `resa`: yield;
- `copertura`: coverage;
- `fedeltà`: fidelity.

## Changes in v6 (after pilot 1)

- 3 repetitions with majority vote. Instability is the share of repetition pairs whose outcomes disagree.
- Gate criterion 4a is now `early_miss`: wrong attributions caused by a false alarm before the injected point. Criterion 5 is computed on the baseline only.
- Improved support checker:
  - a full stop after a number is no longer read as part of it;
  - labels such as "Step 2" and "Agent 1" are not counted as numbers;
  - verbal results ("yielding") are recognised;
  - chain intermediates, fractions, and skipping of algebra.
- Tasks from pilot 1 are excluded (`data/used_ids.txt`).

## HotpotQA

1. Download `hotpot_dev_distractor_v1.json` from the official README (github.com/hotpotqa/hotpot, section "Data Download").
2. Put it in `data/`.
3. Run with `--data hotpot:data/hotpot_dev_distractor_v1.json`.

Only `bridge` questions are used by default.

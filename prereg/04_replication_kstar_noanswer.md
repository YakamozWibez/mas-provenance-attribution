# Pre-registration 04: replication, filtered ambiguity, and attribution without the correct answer

Written on 26 November 2026. It is committed **before** the data it describes exist. The 400 problems used here have not been run by any model in this project.

## Why this pre-registration

Pre-registration 03 left three questions open:
1. Whether trace ambiguity $k$ predicts **how much** declared sources help. The result was inconclusive because of too few pairs.
2. Whether $k$ measures ambiguity well. $k$ counts identical-looking elements, but a careful reader can use other cues, such as element order.
3. Whether the results hold when the judge is **not** given the correct answer, which is the more realistic audit setting.

It also showed an unplanned performance cost of combining structure and sources (−0.12 [−0.20, −0.04]). That observation is tested here as a pre-registered hypothesis.

## Disclosures (what was seen before writing this file)

- The distribution of $k^*$ (defined below) was inspected on the pilot-2 and pilot-3 logs, **without** joining it to the judge's answers. In pilot 3, for M errors without sources, the order cue resolved most look-alike cases and cleared the true culprit in 8 of 80.
- While testing the analysis script on the pilot-3 data, the predicted effect of sources from $k^*$ was printed: +0.054, against +0.256 from $k$. The observed pilot-3 effect (+0.196) was already known. The expectation behind H3 below is therefore **informed by pilot 3**. That is why H3 is tested on new data, with both outcomes stated in advance.
- The performance-cost hypothesis (H6) comes from an exploratory comparison in pilot 3.

## Data and procedure

1. **Tasks.** 400 new GSM8K test problems (seed 20261126). The 200 problems of pilots 1–3 are excluded (`code/used_ids_p1_p2_p3.txt`).
2. **Agent run.** Same model (Qwen/Qwen3.5-9B on Together, temperature 0, thinking off), same prompts (hash `7f09f833c18153cf`), same pipeline (v6), 3 repetitions with majority vote, the same four conditions, U and M errors. Config: `code/config_p4.json`. No gate.
3. **Prediction file.** After the agent run and **before** any judge call:
   - compute $k$ (`ambiguity.py`) and $k^*$ (`ambiguity_star.py`);
   - run `analyze_04.py predict`;
   - commit the outputs.
4. **Judge.** `openai/gpt-oss-120b`, same prompt and settings as before, four runs:
   - full traces and stripped traces;
   - each with and without the correct answer (`judge.py --no-answer`).
5. **Test.** `analyze_04.py test`.

## Definitions

- $k$: as in pre-registration 03. The number of elements of agents 1–2 that look identical to the injected element in the stripped view (same numbers and arithmetic symbols, same declared sources). The predicted bound is $\frac{1}{k}$.
- $k^*_B$ (`code/ambiguity_star.py`): the look-alike candidates that remain after two tests.
  - **Test A, arithmetic.** A candidate containing a consistent computation "expression = value" is cleared.
  - **Test B, order.** Agent 1's numbers are aligned in order with the task's numbers (digits and number words). Global alignment: match +2, mismatch −1, gap −1, with ties resolved towards the diagonal. A candidate of agent 1 whose numbers all fall on equal aligned positions is cleared.

  Clearing is defined **relative to the injected error types**: one value changed, fabricated (U) or swapped (M). It is not evidence that a step is correct in general. A step with consistent arithmetic can still be a reasoning error, and we do not inject those.
- Bound for a reader that trusts the order cue: $\frac{1}{k^*_B}$. If the order cue clears the true culprit, the bound is set to 0.
- Predicted effect of sources from either measure: the bound with sources minus the bound without sources, per task and structure. This is a heuristic, as in pre-registration 03.

## Hypotheses

All hypotheses concern M errors and the stripped view unless stated otherwise. Pairs are tasks with a decisive M error both with and without sources, within the same structure, with the two structures pooled. Intervals are 95% bootstrap CIs (5,000 resamples, seed 4).

### Primary

- **H1, replication.** Sources help, with the correct answer shown: the CI lower bound of the effect is above 0.
- **H2, $k$ predicts the size.** Let $D$ be the observed effect minus the effect predicted from $k$. The verdict is:
  - **supported** if the CI of $D$ lies inside $[-0.20, +0.20]$;
  - **contradicted** if the CI excludes 0;
  - **inconclusive** otherwise.
- **H3, informational or accessibility.** Let $D^*$ be the observed effect minus the effect predicted from $k^*_B$. The two outcomes are stated in advance:
  - **A, informational.** The CI of $D^*$ lies inside $[-0.20, +0.20]$: sources help about as much as the information they add, for a reader using the order cue.
  - **B, accessibility.** The CI lower bound of $D^*$ is above 0: sources help the judge **more** than the information they add. They make information that is already in the trace (through order) accessible and reliable.
  - Otherwise: neither, or inconclusive. Reported as such.

### Secondary

- **H4.** Among stripped cases without sources that have look-alikes ($k \ge 2$), the judge is right more often when the order cue resolves the case ($k^*_B = 1$, culprit not cleared) than when it does not. CI lower bound above 0. Tested only if each group has at least 5 cases.
- **H5, no correct answer.** On stripped traces, the effect of sources is larger without the correct answer than with it, i.e. the difference of the two effects has its CI lower bound above 0. Also reported:
  - the change in $J$ when the answer is removed (no sources);
  - the effect of sources on full traces without the answer.
- **H6, performance cost.** $P$(structured + sources) − $P$(free), paired over the 400 tasks: CI upper bound below 0. The other two conditions are reported descriptively, together with token use.

Six tests are run, with no correction for multiplicity; H1–H3 are primary. U errors are reported descriptively.

## Limitations stated in advance

- One agent model, one judge, one task family.
- Errors are injected value errors, almost always at agent 1; reasoning errors are not covered.
- Stripping removes words artificially.
- $k^*_B$ uses only two cues: arithmetic and order. A stronger reader could use others.
- The judge is told that the final answer is wrong.

## Budget

- Agent run: at most 6 USD (estimate about 2.2).
- Judges: at most 2 USD each (estimate about 1 each).

## Commands (after this file is committed)

```
python run_pilot.py --family math --data gsm8k:data/gsm8k_test.jsonl --llm openai --model Qwen/Qwen3.5-9B --config config_p4.json --out results_p4
python ambiguity.py      results_p4/math_gsm8k_Qwen-Qwen3.5-9B.jsonl > results_p4/ambiguity.jsonl
python ambiguity_star.py results_p4/math_gsm8k_Qwen-Qwen3.5-9B.jsonl > results_p4/ambiguity_star.jsonl
python analyze_04.py predict results_p4/ambiguity.jsonl results_p4/ambiguity_star.jsonl > results_p4/prediction_04.txt
# commit the prediction, then the four judge runs:
python judge.py --results results_p4/math_gsm8k_Qwen-Qwen3.5-9B.jsonl --data gsm8k:data/gsm8k_test.jsonl --model openai/gpt-oss-120b --price-in 0.2 --price-out 0.8 --max-tokens 4000 [--strip] [--no-answer]
python analyze_04.py test results_p4/math_gsm8k_Qwen-Qwen3.5-9B.jsonl results_p4/ambiguity.jsonl results_p4/ambiguity_star.jsonl \
   results_p4/..._judge_openai-gpt-oss-120b.jsonl results_p4/..._strip.jsonl results_p4/..._noans.jsonl results_p4/..._strip_noans.jsonl
```

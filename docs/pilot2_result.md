# Pilot 2 result: declared sources do not help a competent attributor on short traces

Research line 2: attributability and performance in LLM multi-agent systems.
Author: Emme (independent researcher). 18 November 2026. Exploratory result, not peer reviewed.

## In one line

In a 3-agent chain solving GSM8K problems, making agents declare their sources does **not** improve error attribution by a competent LLM judge. The judge finds the right element about 93–100% of the time, with or without sources. Declaring sources does **cost** 51–78% more tokens. The large benefit of sources that our deterministic rule measured was an **artefact of the rule**.

## Question

At equal task and budget, what happens when agents are forced to use structured messages and to declare their sources?

1. How much does error attribution improve (taking into account that declared sources may be false), and for which kind of error?
2. Does task performance improve, stay the same or get worse?

## Design (summary)

- **Chain:** planner, calculator, checker. Only the first agent sees the task text.
- **Conditions (2 × 2):** free or structured messages, × with or without declared sources.
- **Agent model:** Qwen/Qwen3.5-9B on Together, temperature 0, thinking disabled, 3 repetitions with majority vote.
- **Injected errors**, kept only when decisive (they change the final answer):
  - **U**: a fabricated value;
  - **M**: a misbinding, i.e. a true value taken from another sentence of the task.
- **Attribution:**
  - *rule*: deterministic, returns the first element not supported by its allowed sources;
  - *judge*: an LLM (`openai/gpt-oss-120b`), one call per case, all-at-once as in Who&When. It reads the task, the messages with their element ids, any declared sources, the wrong answer and the correct answer.
- **Tasks:** 50 new GSM8K problems (seed 20261118). The 50 problems of pilot 1 are excluded.
- **Fixed before seeing the data:** protocol, thresholds and how the judge would be read. See `docs/protocol_pilot2_IT.pdf` (Italian, the original pre-run document) and `prereg/01_pilot2.md`.

## Results

### Pilot gate: STOP on criterion 1

| Criterion | Threshold | Value | Outcome |
|---|---|---|---|
| 1 No ceiling or floor | $0.2 \le P_{\text{base}} \le 0.8$ | 0.92 | **failed** |
| 2 Decisive errors | yield ≥ 0.5 | 0.54–0.93 | passed |
| 3 Source coverage | ≥ 0.7 | 0.83 / 0.98 | passed |
| 4a False alarms that shift attribution | ≤ 0.10 | 0–0.088 | passed |
| 5 Baseline instability | ≤ 0.10 | 0.027 | passed |
| 6 Tasks required | ≤ 1,200 | ~480 | passed |

### Performance (question 2)

| Condition | $P$ (majority) | tokens per task | tokens vs baseline |
|---|---|---|---|
| free | 0.92 | 1,062 | — |
| free + sources | 0.90 | 1,608 | +51% |
| structured | 0.88 | 1,603 | +51% |
| structured + sources | 0.88 | 1,892 | +78% |

With 50 tasks, the differences in $P$ (2–4 points) cannot be told apart from noise, and the ceiling also limits them. The token cost, by contrast, is large and consistent.

### Attribution (question 1): correct element

| Condition | U rule | U judge | M rule | M judge |
|---|---|---|---|---|
| free | 1.000 | 0.974 | 0.273 | 0.909 |
| free + sources | 1.000 | 1.000 | 0.926 | 0.889 |
| structured | 0.900 | 1.000 | 0.000 | 0.964 |
| structured + sources | 0.905 | 0.976 | 0.900 | 0.933 |

Effect of sources on the judge (with sources minus without, paired by task, 95% bootstrap CI):

| | free | structured |
|---|---|---|
| U | +0.030 [+0.000, +0.091] | −0.027 [−0.081, +0.000] |
| M | +0.000 [−0.375, +0.375] (8 tasks) | +0.000 [+0.000, +0.000] (23 tasks) |

**Reading fixed in the protocol before the data:** if, for M, the interval includes 0 in both the free and the structured condition, then the benefit of sources seen with the rule is an artefact of the rule. **That is the case.** Without sources, the rule finds the right element for M errors in 8% of cases; the judge does in 95%.

## Interpretation

- **The rule is blind to M errors without sources, by construction.** The swapped value is true and appears in the task, so nothing in the trace marks it as unsupported. The judge catches the swap because it compares the message with the task text.
- **In this design the trace determines the cause of the error.** The chain is short, the error can be checked against the task, and the judge knows the correct answer. Under these conditions declared provenance adds no information, so its value is zero.
- **Link to the author's theorem.** The theorem concerns the non-injectivity of the map from decisions to observable state. It leads us to expect provenance to matter only when different causes produce the same observed trace. This pilot is the injective case, and the null effect is consistent with that expectation. Testing the non-injective case is the next step, pre-registered separately (`prereg/02_...`).

## Limitations

- One agent model (9B), one task family (GSM8K), 50 problems.
- Errors are injected, not natural.
- The judge is given the correct answer, which makes attribution easier.
- Few paired tasks for M in the free condition (8).
- The manual check of 20 cases (criterion 4b) has not been done yet.
- Criteria 4 and 5 were redefined after pilot 1. This was declared in the protocol, and pilot 2 uses new problems.

## Costs

Pilot 1 about 0.20 USD; pilot 2 about 0.27 USD; judge about 0.10 USD.

## Reproduction

See `README.md`. Raw data is in `results/`: one jsonl per run, the report, the manual-check sample and the judge decisions.

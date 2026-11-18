# Pre-registration 01: pilot 2

Fixed on 18 November 2026, before running pilot 2. The full text, with the reasons for each choice, is in `docs/protocol_pilot2_IT.pdf` (Italian). This file summarises it for the repository history.

**Note on dates.** The repository was opened after pilot 2. For this file, the trustworthy evidence of the pre-run date is the protocol PDF, which was written before the run. Later pre-registrations (02 onward) are committed **before** the analyses they describe are run.

## Changes from pilot 1 (declared)

- 3 repetitions with majority vote.
- Criterion 4a redefined as `early_miss`: wrong attributions caused by a false alarm before the injected point.
- Criterion 5 computed on the baseline condition only.
- Improved support checker (v6).
- 50 new problems; pilot-1 problems are excluded.

## Gate

| # | Criterion | Threshold |
|---|---|---|
| 1 | Majority $P$ in the free condition | between 0.20 and 0.80 |
| 2 | Yield of decisive errors, for U and M, in every condition | ≥ 0.50 |
| 3 | Source coverage in the conditions with sources | ≥ 0.70 |
| 4a | `early_miss` in every condition | ≤ 0.10 |
| 4b | Rule/manual agreement on 20 cases | ≥ 19/20 |
| 5 | Instability of the free condition | ≤ 0.10 |
| 6 | Tasks required by $\psi$ with $\Delta = 0.05$ | ≤ 1,200 per family |

Validity condition: invalid outputs < 0.10. The rules are not to be changed to make the pilot pass.

## LLM judge (added before seeing the pilot-2 results)

- **Model:** the first non-Qwen model available, taken from a fixed priority list. DeepSeek-V3.1 and the other candidates were not serverless, so the judge is `openai/gpt-oss-120b`.
- **Primary outcome for question 1:** the effect of sources on $J_M$, paired by task, with a 95% bootstrap CI.
- **Reading:** if the interval includes 0 in both the free and the structured condition, the benefit of sources on M seen with the rule is an artefact of the rule.

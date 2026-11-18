# Pre-registration 02: does provenance help when the trace stops determining the cause?

Written on 18 November 2026. It is committed **before** the analysis it describes is run. The judge has not yet seen any stripped trace.

## Background

In pilot 2 (`docs/pilot2_result.md`), a competent LLM judge attributed injected errors almost perfectly with or without declared sources. In that design the trace **determines** the cause of the error, for two reasons:
- agents describe each value in words ("6 pounds of onions");
- the judge can compare those words with the task.

## Prediction from the theorem

The author's theorem says attribution is limited when the map from causes to the observable trace is not injective. The prediction we derive from it is that declared provenance is useful **only to the extent** that the trace has lost the information needed to identify the cause:

$$\Delta_{\text{sources}} \approx 0 \quad \text{when the trace determines the cause}$$

$$\Delta_{\text{sources}} > 0 \quad \text{when it does not}$$

## Manipulation

We use the same pilot-2 logs, the same judge (`openai/gpt-oss-120b`), the same prompt and settings, with one change. In every agent message, each element is reduced to its **numbers and arithmetic symbols only**: every word is removed, and so are the reasoning paragraphs (`M*.R`). The task text (M0), the element ids, the declared sources (in the conditions with sources), the wrong final answer and the correct answer remain visible. Implementation: `judge.py --strip`.

After stripping, a misbinding (M) such as "6 pounds of onions" becomes "6". The judge can no longer read which quantity the agent meant. Declared sources such as "6 (declared sources: M0.E1)" still carry that binding.

## Hypotheses (for M errors; U is reported descriptively)

Let $J$ be the share of cases where the judge names the correct element. The effect of sources is

$$\Delta_{\text{sources}} = J_{\text{with sources}} - J_{\text{without sources}}$$

paired by task, within the same message structure (free or structured), with the two structures pooled. All intervals are 95% bootstrap CIs over pairs (5,000 resamples, seed 2).

- **H1 (manipulation check).** Without sources, stripping lowers $J_M$ by at least 0.20, and the drop is reliable. Both conditions must hold for $J^{\text{strip}}_M - J^{\text{full}}_M$: the point estimate is ≤ −0.20 and the CI upper bound is < 0.
- **H2 (main).** On stripped traces, $\Delta_{\text{sources}} > 0$ for M: the lower bound of the CI is above 0.
- **H3 (descriptive).** Interaction: $\Delta^{\text{strip}}_{\text{sources}} - \Delta^{\text{full}}_{\text{sources}}$, with its CI.

## Decision rule (fixed now)

| H1 | H2 | Conclusion |
|---|---|---|
| fails | — | **Inconclusive.** Stripping did not remove enough information. The stripping is not modified on these data. |
| holds | holds | **Supported.** Provenance helps when the trace no longer determines the cause. |
| holds | fails | **Not supported** in this setting. |

## Known limitations, stated in advance

- About 30 M pairs, so this is a rough estimate.
- Even without words, a strong judge may recompute the problem and spot the odd number. This works against H1, and H1 is there to detect it.
- Labels are removed artificially. This mimics agents that exchange bare values (for example JSON fields without descriptions); it is not natural behaviour.
- The data are the pilot-2 logs, whose gate stopped on criterion 1 (ceiling on $P$). This does not affect question 1.

## Commands (after this file is committed)

```
python judge.py --results results/math_gsm8k_Qwen-Qwen3.5-9B.jsonl --data gsm8k:data/gsm8k_test.jsonl \
                --model openai/gpt-oss-120b --price-in 0.2 --price-out 0.8 --max-tokens 4000 --strip
python analyze_02.py results/math_gsm8k_Qwen-Qwen3.5-9B_judge_openai-gpt-oss-120b.jsonl \
                     results/math_gsm8k_Qwen-Qwen3.5-9B_judge_openai-gpt-oss-120b_strip.jsonl
```

Budget: at most 1 USD.

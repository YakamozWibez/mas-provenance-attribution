# Results of pre-registrations 02 and 03: when does declared provenance help attribution?

Author: Emme (independent researcher). 24 November 2026. Exploratory work in progress, not peer reviewed.

## Summary

- **Full traces.** When the agents' messages say in words what each value is, a competent LLM judge finds injected errors almost always (89–100% across conditions in pilots 2 and 3), with or without declared sources. Sources add little or nothing.
- **Stripped traces.** When we remove the words and keep only the numbers, the judge's accuracy on misbinding errors (M) collapses. Declared sources recover part of the loss. This was pre-registered (02) and replicated on 100 new problems (03).
- **Trace ambiguity.** We define $k$, the number of candidate elements the trace cannot tell apart, and compute it before any judge call. Cases with $k \ge 2$ are missed far more often than cases with $k = 1$ (pre-registered, confirmed).
- **Size of the effect.** Whether $k$ also predicts **how much** sources help was pre-registered with an equivalence margin of ±0.20. The result is **inconclusive**: the observed effect is +0.196, the predicted one +0.256, and the interval of the difference is [−0.217, +0.095].

## 1. Setup in brief

| Item | Value |
|---|---|
| Chain | 3 agents (planner, calculator, checker); only agent 1 sees the task |
| Conditions | free or structured messages × with or without declared sources |
| Agent model | Qwen/Qwen3.5-9B (Together), temperature 0, thinking off, 3 repetitions with majority vote |
| Injected errors | **U**: fabricated value; **M**: misbinding, i.e. a true value from another task sentence. Kept only if decisive (the final answer changes) |
| Attributor | LLM judge `openai/gpt-oss-120b`, one call, "all at once" (as in Who&When). It sees the task, the element ids, any declared sources, the wrong answer and the correct one |
| Stripped view | every word removed from agent messages; numbers and arithmetic symbols kept; task, ids and declared sources unchanged |
| $J$ | share of cases where the judge names the exact element where the error was introduced |

See `README.md` and `docs/pilot2_result.md` for the full design.

## 2. Pre-registration 02 (pilot-2 logs, 50 problems)

Pre-registered in `prereg/02_non_injective.md` before the stripped judge run. Results are for M errors.

| Hypothesis | Threshold | Result | Outcome |
|---|---|---|---|
| H1: stripping lowers $J$ without sources | ≤ −0.20, CI upper < 0 | −0.615 [−0.769, −0.462] (0.95 → 0.33) | holds |
| H2: sources help on stripped traces | CI lower > 0 | +0.355 [+0.161, +0.548], 31 pairs | holds |

For U errors the effect is small (+0.06; the interval touches 0). This is expected, because a fabricated value is detectable without words.

## 3. Pre-registration 03 (100 new problems)

Pre-registered in `prereg/03_ambiguity_bound.md` before any of these data existed. The prediction (`results/pilot3/prediction_03.txt`) was committed **after** the agent run and **before** any judge call.

### Definition

For an injected error, $k$ is the number of elements in agents 1–2 that look identical to the injected element in the view the attributor gets: same text (or the same numbers, when stripped) and the same declared sources. If the $k$ candidates are **equally likely** to be the cause, given the trace, the best possible accuracy is

$$J_{\max} = \frac{1}{k}$$

**Correction to the pre-registration text.** `prereg/03` says that an attributor is right "at most" $\frac{1}{k}$. That holds only under the equal-likelihood assumption. If other cues in the trace (for example element order or arithmetic consistency) make one candidate more likely, the best possible accuracy can be higher than $\frac{1}{k}$. H1 was designed to detect whether the judge exceeds the bound. On average it did not: its accuracy stayed well below $\frac{1}{k}$. This does not show that such cues are absent, only that the judge did not exploit them enough to beat the bound. The pre-registration file is left unchanged.

The predicted effect of sources, per task and structure, is the following **heuristic** (it is a modelling choice, not a derived result):

$$\hat{\Delta} = \frac{1}{k_{\text{with}}} - \frac{1}{k_{\text{without}}}$$

### Results (M errors, stripped view, two structures pooled, 56 pairs)

| Hypothesis | Rule | Result | Outcome |
|---|---|---|---|
| H2b: sources help (replication of 02) | CI lower > 0 | +0.196 [+0.036, +0.357] | **replicated** |
| H2: $k$ predicts the size, $D = \Delta - \hat{\Delta}$ | CI inside [−0.20, +0.20] | $\hat{\Delta}$ = +0.256; $D$ = −0.060 [−0.217, +0.095] | **inconclusive** |
| H1: judge does not beat $\frac{1}{k}$ (no sources) | violated if CI lower > 0 | $J$ = 0.362 vs bound 0.752 | bound respected, but loose |
| H3: $J(k = 1) > J(k \ge 2)$ (no sources) | CI lower > 0 | +0.257 [+0.051, +0.459] (41 vs 39 cases) | **confirmed** |

### Descriptive

| Quantity | Value |
|---|---|
| Effect of sources on full traces (predicted +0.009) | +0.054 [+0.000, +0.125] |
| $J$ with sources, stripped (free / structured) | 0.566 / 0.620, against a bound of 1.00 / 0.99 |
| U errors: observed vs predicted effect, stripped | +0.038 vs +0.023 |

### Exploratory (not pre-registered as separate tests)

| Structure | Pairs | Predicted | Observed |
|---|---|---|---|
| structured | 36 | +0.310 | +0.306 |
| free | 20 | +0.158 | +0.000 |

### Agent run (reported; no gate was pre-registered for this run)

| Condition | $P$ | tokens per task |
|---|---|---|
| free | 0.88 | 1,125 |
| free + sources | 0.90 | 1,644 (+46%) |
| structured | 0.83 | 1,584 (+41%) |
| structured + sources | 0.76 | 1,950 (+73%) |

Baseline instability across repetitions was 0.133, above the 0.10 used as a pilot criterion in earlier runs.

Paired differences in $P$ against the free condition (same 100 tasks, 95% bootstrap CI, descriptive and not corrected for multiple comparisons): free + sources +0.02 [−0.06, +0.10]; structured −0.05 [−0.15, +0.04]; structured + sources −0.12 [−0.20, −0.04]. Only the last interval excludes 0. This hints at a performance cost of combining structure and sources, but it was not a pre-registered test.

## 4. Interpretation

Let $C$ be the true cause (the element where the error was introduced), $T$ the trace (everything the attributor reads: the task, the agent messages with element ids, the wrong answer and the correct one), $S$ the declared sources, and $\hat{C}$ the attributor's answer. The attributor sees only the trace (plus, at most, its own independent randomness), so $C \to T \to \hat{C}$ is a Markov chain. Two standard facts then apply.

1. **No attributor creates information** (data processing inequality):

$$I(C;\hat{C}) \le I(C;T)$$

2. **Sources can add at most what the trace is missing.** This is a general property of conditional mutual information:

$$I(C;S \mid T) \le H(C \mid T)$$

To connect this with $k$ an **extra assumption** is needed. Suppose that, given a trace $t$, the cause can only be one of the $k(t)$ identical-looking candidates. Then

$$H(C \mid T = t) \le \log_2 k(t)$$

and, averaging over traces,

$$H(C \mid T) \le \mathbb{E}\left[\log_2 k(T)\right]$$

This assumption is **not verified**. $k$ is computed from a simple signature (identical text, or identical numbers, plus sources). It is not derived from the true conditional distribution of $C$ given $T$.

**Reading of the results within this frame:**
- With full traces the cause is essentially determined ($k = 1$ in almost all cases, so $H(C \mid T) \approx 0$ under the assumption above), and sources added almost nothing, as observed.
- With stripped traces, many cases have $k \ge 2$, sources have room to help, and they did.

This is consistent with the non-injectivity argument in the author's earlier theoretical work. It illustrates that argument; it does not test it formally.

$k$ identifies **which** cases are hard (H3). Whether it predicts **how much** sources help (H2) remains open. One possible reason is that the judge does not reach the bound: without sources its accuracy is about half of $\frac{1}{k}$ on average, and with sources about 0.6 of it.

The deterministic rule reaches 0.86–0.94 on M errors in the conditions with sources. This number is **not directly comparable**: the rule reads the full text, not the stripped one, and checks numbers against declared sources mechanically. It is consistent with the idea that LLM attributors read provenance rather than verify it, a pattern also reported by Arslan (2026) for LLM audit layers, but a like-for-like test is still needed (next steps, point 2).

## 5. Limitations

- One agent model, one judge, one task family (GSM8K).
- Errors are injected, almost always at agent 1, and are value errors (a wrong number) that can be checked against the task. Errors of reasoning or judgement, where no reference exists, are not covered.
- The experiment concerns provenance of **information** (where each value came from), not provenance of **decisions** (why a step was taken). The latter is what the author's earlier theoretical work refers to as deliberative provenance.
- Words are removed artificially. Whether meaning is lost this way in real systems was not measured.
- The judge knows that exactly one error exists and is given the correct answer, which makes its task easier than a real audit.
- 56 pairs for the main test. $k$ ignores element order and arithmetic consistency.
- The bound $\frac{1}{k}$ assumes the $k$ candidates are equally likely; the link between $k$ and $H(C \mid T)$ rests on an unverified assumption (section 4).
- The idea behind $k$ arose after pilot 2. That is why it was tested on new data, with the prediction committed before the judge ran.
- Declared provenance is not a new idea: citations in retrieval-augmented generation, execution provenance in agent frameworks (see the 2026 survey *From Agent Traces to Trust*), post-hoc reconstruction of information use (BLADE, 2026) and agent-declared suspected origins (Arslan, 2026). The contribution here is a measurement of **when** agent-declared provenance helps attribution.

## 6. Next steps

1. **Pre-registration 04**, three parts only:
   - about 250 new problems with the same rules, to settle H2 and to test the performance cost of structure plus sources (section 3);
   - a second ambiguity measure, $k^*$: the candidates that an ideal reasoner could not exclude using numbers, arithmetic, element order, the task and the correct answer. Both outcomes will be stated in advance: either $k^*$ predicts the judge better than $k$, or $k^* \approx 1$ while sources still help the judge. The second outcome would mean that sources make existing information **accessible** rather than adding information;
   - the judge without the correct answer, on the existing pilot-3 logs.
2. Natural rather than artificial loss of meaning: first agents asked to write compact messages to save tokens, then summaries and longer chains on multi-hop questions.
3. Test whether instructing the judge to verify sources closes the gap to the mechanical check.
4. Manual check of 20 cases (criterion 4b), still pending.

## Costs

Pilot 3 agent run 0.54 USD; judges 0.22 + 0.26 USD. Total for the project so far: about 1.7 USD.

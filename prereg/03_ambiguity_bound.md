# Pre-registration 03: an ambiguity bound predicts the value of declared sources

Written on 23 November 2026. Committed **before** the data it describes exist. The 100 problems used here have not been run by any model in this project.

## Idea

The author's theorem says attribution fails when different causes leave the same observable trace (the map from causes to traces is not injective). We turn this into a number that can be computed **from the trace alone**, before any attributor looks at it.

For an injected error, let $k$ be the number of elements in the agent messages (agents 1 and 2; reasoning paragraphs and final answers excluded) that look **identical** to the injected element in the view the attributor receives. The injected element itself counts, so $k \ge 1$.

- **Full view:** identical means the same text (lowercased) and the same declared sources.
- **Stripped view** (as in pre-registration 02): identical means the same numbers and arithmetic symbols, and the same declared sources.

Declared sources exist only in the +sources conditions. The implementation is `code/ambiguity.py`, committed with this file. Order and position of elements are not part of $k$.

An attributor that cannot tell the $k$ candidates apart is right at most

$$J_{\max} = \frac{1}{k}$$

of the time. Declared sources can only lower $k$. The **predicted effect of sources** for one task and one message structure is therefore

$$\hat{\Delta} = \frac{1}{k_{\text{with sources}}} - \frac{1}{k_{\text{without sources}}}$$

averaged over tasks.

## Exploratory origin (declared)

The idea came after pilot 2. On the pilot-2 data (stripped view, M errors) the prediction is $+0.403$ and the observed effect is $+0.355$. Those judge results were already known when $k$ was defined, so this is **not** evidence. It is the reason for this test on new data.

## Data and procedure

1. **Tasks:** 100 new GSM8K test problems (seed 20261123). The 100 problems of pilots 1 and 2 are excluded (`code/used_ids_p1_p2.txt`).
2. **Run:** same model (Qwen/Qwen3.5-9B on Together, temperature 0, thinking off), same prompts (hash `7f09f833c18153cf`), same pipeline (v6), 3 repetitions with majority vote, the same four conditions, U and M errors. Config: `code/config_p3.json`. No gate: this run serves attribution only, and $P$ is reported.
3. **Prediction file:** after the run and **before** any judge call, compute $k$ (`ambiguity.py`) and the predictions (`analyze_03.py predict`), and commit them.
4. **Judge:** `openai/gpt-oss-120b`, same prompt and settings as before, on full and on stripped traces.
5. **Test:** `analyze_03.py test`.

## Hypotheses (M errors, stripped view; U reported in the same way, descriptively)

All intervals are 95% bootstrap CIs over pairs or cases (5,000 resamples, seed 3). Pairs are tasks with a decisive M error in both the condition with sources and the one without, within the same structure; the two structures are pooled.

- **H2b (replication of pre-registration 02).** The observed effect of sources is positive: the lower bound of the CI is above 0.
- **H2 (main: $k$ predicts the size of the effect).** Let $D$ be the observed effect minus the predicted effect $\hat{\Delta}$, per pair.
  - **Supported** if the CI of $D$ lies entirely inside $[-0.20, +0.20]$.
  - **Contradicted** if the CI of $D$ excludes 0.
  - **Inconclusive** otherwise.
- **H1 (bound).** Without sources, in the stripped view, the judge does not beat the bound. **Violated** if the CI of $J - \frac{1}{k}$ has a lower bound above 0. A violation would mean the trace carries information that $k$ ignores, such as element order.
- **H3 (which cases are missed).** Without sources, in the stripped view, $J$ is higher where $k = 1$ than where $k \ge 2$ (CI lower bound above 0). This is tested only if both groups have at least 5 cases.

**Descriptive:**
- the efficiency of the judge with sources, $J$ against the bound;
- the effect of sources on full traces, where the prediction is 0.

## What each outcome means

| H2 | Reading |
|---|---|
| supported | The value of declared sources is predicted, within ±0.20, by how much they reduce trace ambiguity. This is a quantitative consequence of non-injectivity. |
| contradicted | Ambiguity as defined here does not predict the size of the effect. |
| inconclusive | Not enough precision; reported as such. |

## Limitations stated in advance

- $k$ counts identical-looking elements only. Position and arithmetic consistency are ignored, so the bound may be loose (H1 checks this).
- One agent model, one judge, one task family.
- Stripping removes words artificially.

## Budget

- Run: at most 3 USD (estimate about 0.8).
- Judges: at most 1 USD each.

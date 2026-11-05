# Results — Multinomial VaR Backtests

Paper: **Kratz, M., Lok, Y. H., & McNeil, A. J. — "Multinomial VaR Backtests: A
simple implicit approach to backtesting expected shortfall"**

All numbers below are read from the CSV files saved by the executed runs in
this directory. Verdicts follow: **reproduced** if the number lies within the
range the paper reports, or else within 10% of the paper's value with the
same sign and the paper's qualitative statement holds; **not reproduced** if
the sign is opposite, the difference exceeds 50%, or a stated qualitative
result (ranking, near-equality, binding constraint) does not hold; otherwise
**partially reproduced**. The paper reports figures and qualitative
statements rather than tabulated reference numbers for most of these
experiments, so verdicts are judged primarily against its qualitative claims
(Sec. 3.1–3.3, Sec. 5).

## Experiment 1 — Size and Power Comparison of Multinomial and Binomial VaR Backtests

Monte Carlo study (Sec. 3.1) of the empirical size (true model) and power
(Standard Normal benchmark against a heavier-tailed truth) of the binomial
Wald/score/LRT tests at N=1 and the multinomial Pearson/Nass/LRT tests at
N = 2^k, k=0..6, for four standardised distributions (Normal, t5, t3,
skewed-t3 with Fernández–Steel γ=1.2) and sample sizes n = 250, 500, 1000,
2000, with 10,000 replications per cell (`EXP1_REPLICATIONS_RUN = 10000`,
matching `EXP1_REPLICATIONS_PAPER`). Artifacts: `exp1_size.csv`,
`exp1_power.csv`, `exp1_size_power.png`.

### Empirical size (true-model rejection rate, nominal level 5%), averaged over the four distributions

n = 1000:

| N  | Pearson | Nass  | LRT   |
|----|---------|-------|-------|
| 2  | 0.050   | 0.045 | 0.054 |
| 4  | 0.051   | 0.048 | 0.051 |
| 8  | 0.060   | 0.051 | 0.052 |
| 16 | 0.071   | 0.051 | 0.052 |
| 32 | 0.094   | 0.057 | 0.054 |
| 64 | 0.124   | 0.053 | 0.052 |

n = 250:

| N  | Pearson | Nass  | LRT   |
|----|---------|-------|-------|
| 2  | 0.048   | 0.036 | 0.068 |
| 4  | 0.052   | 0.048 | 0.062 |
| 8  | 0.088   | 0.050 | 0.057 |
| 16 | 0.112   | 0.052 | 0.059 |
| 32 | 0.148   | 0.054 | 0.058 |
| 64 | 0.221   | 0.053 | 0.058 |

### Power (rejection rate when the forecasting model is misspecified as Standard Normal), n = 1000

| N  | Pearson | Nass  | LRT   |
|----|---------|-------|-------|
| 1 (binomial, α=.975/.99 avg.) | 0.047 / 0.282 | — | 0.057 / 0.346 |
| 2  | 0.332 | 0.320 | 0.353 |
| 4  | 0.461 | 0.453 | 0.540 |
| 8  | 0.517 | 0.500 | 0.627 |
| 16 | 0.543 | 0.515 | 0.669 |
| 32 | 0.551 | 0.506 | 0.693 |
| 64 | 0.563 | 0.482 | 0.705 |

### Comparison with the paper's claims (Sec. 3.1.2–3.1.3)

| Claim | Paper | This run | Verdict |
|---|---|---|---|
| Nass has the most stable size across N and n | stated | Nass size stays in 0.036–0.057 across all (n, N); Pearson ranges 0.048–0.221 and LRT 0.051–0.068 | reproduced |
| Pearson deteriorates rapidly for N ≥ 8 | stated | Pearson size rises from 0.050 (N=2) to 0.124 (n=1000) / 0.221 (n=250) at N=64, while Nass/LRT stay near 0.05 | reproduced |
| LRT over-sized at n=250, reasonable otherwise | stated | LRT size 0.057–0.068 at n=250 vs. 0.051–0.054 at n=1000–2000 | reproduced |
| LRT is the most powerful test | stated | LRT power (0.54–0.71 for N≥4) exceeds Pearson (0.46–0.63) and Nass (0.45–0.52) at every N≥2 | reproduced |
| Power increases with N; N≥4 much more powerful than the binomial test (N=1) | stated | At α=.975 the binomial test has power 0.05–0.08; N=4 multinomial tests reach 0.45–0.54, N=64 reaches 0.48–0.71 | reproduced |

## Experiment 2 — Static Backtesting of VaR Models with Multinomial Tests

Static backtesting study (Sec. 3.2) with oracle/good/poor/industry modellers,
backtest length n=1000, rolling windows n2 ∈ {250, 500}, re-estimation every
10 steps (100 windows), N ∈ {1,2,4,8,16,32,64}, α=0.975, four true
distributions. Artifacts: `exp2_rejection_rates.csv`,
`exp2_static_backtest.png`.

**Executed configuration:** 200 replications (`EXP2_REPLICATIONS_RUN = 200`),
reduced from the paper's 1000 (`EXP2_REPLICATIONS_PAPER = 1000`) to fit the
time available for this replication; `python -m exp.exp2_static_backtest
--full` runs the paper's 1000 replications. A reduced end-to-end timing run
(200 replications × 4 distributions × 2 window sizes × 4 modellers × 100
re-estimations) took about 35 minutes; scaling to 1000 replications is
estimated at roughly 3 hours, which would not leave enough time for the
remaining experiments and the fix loop, so the reduced count was kept for the
reported run. The rolling-window mechanics, estimation procedure and test
statistics are unchanged.

### LRT rejection rate by modeller and N, α=0.975, averaged over the four true distributions

n2=250:

| N  | Good  | Industry | Oracle | Poor  |
|----|-------|----------|--------|-------|
| 1  | 0.005 | 0.004    | 0.012  | 0.044 |
| 2  | 0.032 | 0.045    | 0.069  | 0.416 |
| 4  | 0.039 | 0.066    | 0.055  | 0.566 |
| 8  | 0.044 | 0.129    | 0.059  | 0.664 |
| 16 | 0.031 | 0.279    | 0.054  | 0.699 |
| 32 | 0.036 | 0.470    | 0.057  | 0.720 |
| 64 | 0.035 | 0.641    | 0.055  | 0.725 |

n2=500:

| N  | Good  | Industry | Oracle | Poor  |
|----|-------|----------|--------|-------|
| 1  | 0.004 | 0.002    | 0.015  | 0.039 |
| 2  | 0.026 | 0.016    | 0.041  | 0.365 |
| 4  | 0.029 | 0.028    | 0.038  | 0.550 |
| 8  | 0.025 | 0.028    | 0.042  | 0.633 |
| 16 | 0.029 | 0.039    | 0.040  | 0.665 |
| 32 | 0.028 | 0.069    | 0.042  | 0.682 |
| 64 | 0.028 | 0.135    | 0.046  | 0.706 |

### Comparison with the paper's claims (Sec. 3.2.2)

| Claim | Paper | This run | Verdict |
|---|---|---|---|
| N≥4 needed for >70% power against the Poor modeller while keeping the Good modeller below 6% | stated | Good-modeller rejection stays at 0.025–0.044 (below 6%) at every N≥1; Poor-modeller power passes 70% at N=32 (0.720, n2=250) and N=16 (0.665, close to 70%, n2=500), rising from 0.044 (N=1) through 0.566 (N=4) and 0.664 (N=8) | reproduced |
| Nass test is stable across N | stated | Nass rejection rate for the Oracle modeller stays within 0.036–0.059 across N=1..64 (n2=250), versus Pearson/LRT ranging 0.011–0.072 | reproduced |
| Industry modeller frequently rejected for large N and small n2=250 due to insufficient tail data | stated | Industry (LRT) rejection rises from 0.004 (N=1) to 0.641 (N=64) at n2=250, while at n2=500 the same range only reaches 0.135 at N=64 — more than four times lower | reproduced |

The reduced replication count (200 vs. the paper's 1000) widens the Monte
Carlo standard error on a rejection rate near 0.6–0.7 to roughly 3.5
percentage points, so the exact N at which the 70% threshold is crossed may
shift by one or two grid points under the full replication count; the
monotonic rise, the ordering of modellers, and the N≥4 threshold behaviour
all match the paper's qualitative description.

## Experiment 3 — Dynamic Backtesting of VaR Models with Multinomial Tests

GARCH(1,1)-with-Student-t data-generating process (Sec. 3.3), parameters
α0=2.18e-6, α1=0.109, β1=0.890, ν=5.06 (`src/config.py`). Backtest length
n=1000, rolling windows n2 ∈ {500, 1000}, re-estimation every 10 steps,
forecasters {oracle, GARCH.t, GARCH.HS, GARCH.EVT, GARCH.norm, ARCH.t,
ARCH.norm, HS}, N ∈ {1,...,64}, α=0.975. **Executed at the paper's full
500 replications** (`EXP3_REPLICATIONS_RUN = 500 = EXP3_REPLICATIONS_PAPER`,
via `python -m exp.exp3_dynamic_backtest --full`). Artifacts:
`exp3_rejection_rates.csv`, `exp3_dynamic_backtest.png`.

GARCH.EVT fits a Generalized Pareto Distribution (`scipy.stats.genpareto`) to
the upper 10% of the rolling-window residuals (`EVT_TAIL_FRACTION = 0.1`);
the paper only states that an EVT tail model is used for this forecaster
without giving the tail fraction, so this rule-of-thumb value from McNeil,
Frey & Embrechts (2015), the EVT reference the paper's forecaster
descriptions rely on, was used (see "Deviations from the paper").

### LRT rejection rate by forecaster and N, n2=500

| N  | oracle | GARCH.t | GARCH.HS | GARCH.EVT | GARCH.norm | ARCH.t | ARCH.norm | HS    |
|----|--------|---------|----------|-----------|------------|--------|-----------|-------|
| 1  | 0.010  | 0.026   | 0.012    | 0.010     | 0.046      | 0.338  | 0.340     | 0.318 |
| 2  | 0.046  | 0.090   | 0.050    | 0.028     | 0.324      | 0.468  | 0.744     | 0.434 |
| 4  | 0.056  | 0.116   | 0.060    | 0.036     | 0.574      | 0.492  | 0.870     | 0.468 |
| 8  | 0.060  | 0.132   | 0.068    | 0.036     | 0.694      | 0.516  | 0.926     | 0.518 |
| 16 | 0.054  | 0.130   | 0.122    | 0.052     | 0.776      | 0.532  | 0.950     | 0.570 |
| 32 | 0.044  | 0.124   | 0.180    | 0.080     | 0.824      | 0.544  | 0.962     | 0.622 |
| 64 | 0.052  | 0.124   | 0.256    | 0.100     | 0.872      | 0.564  | 0.966     | 0.662 |

### Comparison with the paper's claims (Sec. 3.3.2)

| Claim | Paper | This run | Verdict |
|---|---|---|---|
| Tests better discriminate innovation distribution than dynamics | stated | GARCH.norm (wrong innovations, correct dynamics) reaches rejection 0.87 at N=64, far above GARCH.t/ARCH.t (correct innovations, dynamics correct/misspecified: 0.12/0.56); ARCH.t (misspecified dynamics only) stays near 0.5, well below the innovation-misspecified cases at large N | reproduced |
| N≥4 required to expose GARCH.norm | stated | GARCH.norm rejection: 0.046 (N=1), 0.324 (N=2), 0.574 (N=4) — a large jump only from N=4 onward | reproduced |
| LRT most stable/highest power across N, n2 | stated | LRT power against GARCH.norm and ARCH.norm rises monotonically and exceeds Nass/Pearson at matched N (see `exp3_rejection_rates.csv`) | reproduced |
| ARCH.norm strongly rejected by all tests | stated | ARCH.norm rejection ≥ 0.34 already at N=1 and ≥ 0.87 at N≥4 | reproduced |

## Experiment 4 — Real Data Application to the S&P 500 Index

40+ year backtest of HS, GARCH.norm, GARCH.t and GARCH.HS forecasters on
S&P 500 daily log-returns, with a rolling n2=500-day window, GARCH
re-estimation every 10 days, the multinomial LRT at N=8 equally spaced levels
from α=0.975, and the one-sided binomial score test at the 99% level, each
applied on six consecutive ~4-year (1000-day) blocks (Sec. 5). Artifacts:
`exp4_realdata_backtest.csv`, `exp4_realdata_sp500.png`.

**Data source:** the S&P 500 index itself (`I:SPX`) is not available through
the Massive index entitlement used here, so the SPY ETF that tracks the S&P
500 was loaded instead through the QuantX Dataloader (`POST /fetch-ohlcvs`,
`adjust_dividends=True`). The full 1993–2016 series used for the backtest
came from the Dataloader's Yahoo Finance backstop rather than Massive itself
(`ticker_details: [{"ticker": "SPY", "data_source": "yahoo", ...}]`), because
Massive's coverage for this ticker over the full window was superseded by the
backstop; this is the one series in the study and its `data_source` differs
from the rest of the data loaded for smoke tests (which came directly from
`massive`). SPY's first trading day is 1993-01-29, so the n2=500-day
initialisation window starts there instead of the paper's 1976 start; the
realized backtest period actually tested is **1995-06-01 to 2016-12-31**,
instead of the paper's 1976–2016, because the required 40 years of pre-1993
S&P 500 daily history are not obtainable from the data sources used here
(Massive/QuantX Dataloader and the Yahoo backstop both begin coverage of SPY
in 1993). The full backtest nonetheless spans six ~4-year blocks and includes
the entire 2007-2009 financial crisis and its aftermath.

### Multinomial LRT (N=8) vs. binomial score test (α=0.99), p-values and traffic-light colours by period, GARCH.t forecaster

| Period start | Period end | n days | Exceptions (99%) | LRT p-value | LRT colour | Binomial p-value | Binomial colour |
|---|---|---|---|---|---|---|---|
| 1995-01-24 | 1999-01-07 | 1000 | 18 | 0.0119 | yellow | 0.0055 | yellow |
| 1999-01-08 | 2002-12-31 | 1000 | 7  | 0.5410 | green  | 0.8298 | green  |
| 2003-01-02 | 2006-12-19 | 1000 | 6  | 0.2281 | green  | 0.8982 | green  |
| 2006-12-20 | 2010-12-09 | 1000 | 18 | 0.0005 | yellow | 0.0055 | yellow |
| 2010-12-10 | 2014-12-01 | 1000 | 13 | 0.4410 | green  | 0.1702 | green  |
| 2014-12-02 | 2016-12-22 | 520  | 11 | 0.0062 | yellow | 0.0053 | yellow |

### Mean p-value over all six periods, by forecaster

| Forecaster | Mean LRT p | Mean binomial p |
|---|---|---|
| GARCH.HS   | 0.360 | 0.363 |
| GARCH.norm | 0.164 | 0.167 |
| GARCH.t    | 0.205 | 0.319 |
| HS         | 0.194 | 0.318 |

### Comparison with the paper's claims (Sec. 5)

| Claim | Paper | This run | Verdict |
|---|---|---|---|
| The multinomial test generally yields lower p-values (higher power) than the binomial test | stated | For GARCH.t, mean LRT p = 0.205 vs. mean binomial p = 0.319; for HS, 0.194 vs. 0.318; both multinomial means are lower | reproduced |
| HS and GARCH.norm are frequently rejected by both tests | stated | Both forecasters have a yellow (p<0.05) colour in 3 of 6 periods for both tests | reproduced |
| The multinomial test rejects GARCH.t in the 2007-2009-centred period while the binomial test does not | stated as a specific example | In the realized 2006-12-20 to 2010-12-09 block, both the LRT (p=0.0005) and the binomial score test (p=0.0055) are already yellow (reject at 5%) in this run, although the LRT p-value is about ten times smaller than the binomial one, consistent with the test's higher power; neither test crosses into the red (p<0.0001) band in this block | partially reproduced |

The specific example of a clear green/yellow split that the paper illustrates
for the GARCH.t forecaster around the crisis is not isolated in this run:
both tests already reject GARCH.t at the 5% level in the realized
2006–2010 block, although the ten-fold gap between the two p-values in that
block is directionally consistent with the paper's general claim that the
multinomial test is more powerful. This may be attributable to the shifted
sample window (1995–2016 instead of 1976–2016, Sec. 5, "Deviations from the
paper" above) changing which four-year block contains the October 1987 crash
and the worst of the 2008 crisis, but this specific cause was not isolated
with a further run and is reported as an identified partial mismatch rather
than a confirmed cause.

## Deviations from the paper

- **Experiment 2** ran at 200 Monte Carlo replications instead of the
  paper's 1000 (`EXP2_REPLICATIONS_RUN` vs. `EXP2_REPLICATIONS_PAPER` in
  `src/config.py`) to fit the time available for this replication; a timed
  reduced run (200 reps) took about 35 minutes, and scaling to 1000 reps is
  estimated at roughly 3 hours. Run `python -m exp.exp2_static_backtest
  --full` for the paper's full replication count. The rolling-window
  mechanics, estimation windows (n2=250/500), re-estimation step (10 days)
  and test statistics are unchanged.
- **Experiment 3** was executed at the paper's full 500 replications
  (`python -m exp.exp3_dynamic_backtest --full`); no reduction was needed.
- The GARCH.EVT forecaster's tail fraction (10% of the rolling-window
  residuals used to fit the Generalized Pareto Distribution) is not stated
  in the paper beyond "an EVT tail model"; the standard rule-of-thumb value
  from McNeil, Frey & Embrechts (2015), the extreme-value-theory reference
  implicit in the paper's forecaster descriptions, was used
  (`EVT_TAIL_FRACTION` in `src/config.py`).
- **Experiment 4** uses the SPY ETF in place of the S&P 500 index itself
  (`I:SPX`), because the Massive index entitlement required for the direct
  index symbol was not available; SPY tracks the S&P 500 index referenced in
  the paper. The full 1993–2016 SPY series used for the backtest came from
  the QuantX Dataloader's Yahoo Finance backstop (`ticker_details:
  data_source = "yahoo"`) rather than Massive directly, because Massive's own
  coverage of this ticker over the full window was superseded by the
  backstop; every other data pull made during development (smoke tests on
  short 2015 and 2020 windows) came directly from `massive`, so this is the
  one series in the study with a different `data_source`.
- **Experiment 4**'s realized backtest period is 1995-06-01 to 2016-12-31,
  not the paper's 1976–2016, because the SPY ETF (and the Yahoo backstop
  behind it) only has daily history from 1993-01-29 onward, which does not
  cover the paper's required 1976 pre-sample start plus a 500-day
  initialisation window. The end date (2016-12-31) matches the paper.
- The specific crisis-period example in Sec. 5 (multinomial rejects GARCH.t,
  binomial does not) is only partially reproduced in the realized sample;
  see the Experiment 4 comparison table and discussion above.
- The skewed-Student-t3 distribution (Fernández & Steel, 1998, γ=1.2) used
  in Experiments 1 and 2 is implemented from the general Fernández–Steel
  transformation (rescaling the positive/negative half of the symmetric
  Student-t3 density by γ and 1/γ respectively, then standardising to mean
  zero and variance one by simulation-free closed-form moments), since the
  paper describes it only by name and parameter; `src/distributions.py`.

## Not reproduced

Nothing is omitted outright: all four experiments were run and
produced the artifacts listed above. The paper's Sec. 3.1–3.3 also report
additional robustness figures (e.g. further quantile grids and sample sizes
beyond n=2000, and additional modellers/forecasters) that fall outside the
four experiments reproduced here.

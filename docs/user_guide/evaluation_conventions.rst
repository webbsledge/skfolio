.. _evaluation_conventions:

.. currentmodule:: skfolio

**********************
Evaluation Conventions
**********************

A backtest converts a sequence of target weights into a portfolio return series and
summary measures. :class:`~skfolio.portfolio.Portfolio` performs the conversion for
one rebalancing period, :class:`~skfolio.portfolio.MultiPeriodPortfolio` chains the
periods, and :func:`~skfolio.model_selection.cross_val_predict` builds the chain
from a cross-validation scheme. Two conventions control the conversion.

* `weight_drift` selects the weights held between two rebalancing dates: the
  targets throughout the period (default, the constant-weight convention) or
  weights that move with asset prices until the next rebalancing date
  (`weight_drift=True`, the buy-and-hold convention). The default gives the return
  of a constant-mix portfolio [1]_ rebalanced at every observation at no cost. It
  is the quantity the optimizer maximizes. `weight_drift` also sets the holdings
  from which turnover and transaction costs are measured at the next rebalancing date.
* `compounded` selects how the per-period return series is summarized: arithmetic
  sums of returns (default) or products of gross returns, the wealth index
  (`compounded=True`).

The defaults measure **allocation skill**, an expectation-based (ex-ante) evaluation
of the target allocation. `weight_drift=True` with `compounded=True` measures
**realized capital growth**, a path-dependent (ex-post) evaluation along the
historical return path.

A backtest serves one of three goals, and each goal has a quantity to estimate,
the estimand:

* **Expectation (ex-ante)**: the expected return of a strategy, either the expected
  return of the target allocation before transaction costs, which measures the skill
  of the allocation rule, or the expected return of the implemented strategy net of
  transaction costs.
* **Ranking**: the difference in expected performance between candidate
  strategies.
* **Realization (ex-post)**: the wealth path a strategy would have produced on the
  historical sample.

Each convention defines an estimator of the estimand, and the better estimator is
the one with the lower mean squared error, squared bias plus variance. Part 1
treats weight drift, which changes the return series. Part 2 treats compounding,
which changes how the series is summarized. Part 3 treats the timing of
transaction costs. The final section maps each goal to its settings.

Running examples
================

The two strategies below serve as running examples. All magnitudes in this guide
are computed for them, and they lead to different settings for the same goals.

.. list-table::
   :header-rows: 1

   * - Input
     - Equity example
     - Crypto example
   * - Universe
     - 50 large-cap stocks
     - 10 crypto assets
   * - Rebalancing
     - weekly
     - monthly
   * - Portfolio volatility :math:`\sigma_{ann}`
     - 15%
     - 60%
   * - Cross-sectional dispersion :math:`\sigma_d`
     - 1.3% per day
     - 3% per day
   * - Spread of expected returns :math:`s`
     - 5% per year
     - 20% per year
   * - Dispersion of betas to the portfolio :math:`\sigma_\beta`
     - 0.3
     - 0.3
   * - Transaction cost rate :math:`c`
     - 10 bps
     - 20 bps
   * - History :math:`T_{years}`
     - 20 years
     - 5 years

Notation
========

Observations are daily, 252 per year, and annual quantities are quoted in basis
points (1 bp = 0.01%).

* :math:`N`: number of assets. :math:`r_t \in \mathbb{R}^N`: asset linear returns
  over observation :math:`t`.
* :math:`t_k`: the :math:`k`-th rebalancing date. Period :math:`k` covers
  :math:`t_k \le t < t_{k+1}` and contains :math:`n` observations, 5 for weekly
  and 21 for monthly rebalancing.
* :math:`T`: number of observations in the backtest, :math:`T_{years}` its length
  in years.
* :math:`w_k`: target weights set at :math:`t_k`. :math:`u_t`: weights held during
  observation :math:`t`. :math:`\tilde{w}_k`: held weights at the end of period
  :math:`k`, before the trade at :math:`t_{k+1}`. :math:`\| \cdot \|_1`: L1 norm.
* :math:`r^{ptf}_t`: portfolio return of observation :math:`t`. :math:`\mu`,
  :math:`\sigma^2`: its expected value and variance under constant weights,
  :math:`\sigma_{ann} = \sigma \sqrt{252}`.
* :math:`\mu_i`: expected return of asset :math:`i`. :math:`s`: weighted
  cross-sectional standard deviation of the :math:`\mu_i`, in annual units.
* :math:`\beta_i = \operatorname{Cov}(r_i, r^{ptf}) / \sigma^2`: beta of asset
  :math:`i` to the target portfolio. :math:`\sigma_\beta`: weighted cross-sectional
  standard deviation of the :math:`\beta_i`.
* :math:`\sigma_d`: cross-sectional dispersion, the standard deviation of
  :math:`r_{i,t} - r^{ptf}_t` across assets and observations.
* :math:`d_t`: drift term, the drifted minus the constant-weight portfolio return
  of observation :math:`t`. :math:`\bar d`: its expected value averaged over a
  period and annualized. :math:`\gamma_\Delta`: relative change in the variance of
  the return series caused by the drift, of either sign. :math:`\bar{\gamma}_\Delta`:
  upper bound on :math:`|\gamma_\Delta|` to first order.
* :math:`c`: one-off transaction cost rate on the traded notional. :math:`b`:
  transaction cost gap, the annual transaction costs charged under drifted weights
  minus those charged under constant weights.
* :math:`g`: per-period geometric growth rate. :math:`W_T`: wealth index at
  horizon :math:`T`.

Part 1: Weight Drift
====================

Return series under each convention
-----------------------------------

Under constant weights, the return of every observation in period :math:`k` is

.. math:: r^{ptf}_t = w_k \cdot r_t
   \qquad t_k \le t < t_{k+1}

The return is linear in :math:`w_k`, as in the objective of the convex
optimization, so the backtest evaluates the quantity the optimizer maximized.
Within a period the series is permutation invariant: reordering the observations
leaves every measure computed from the empirical distribution of returns
unchanged.

Under drifted weights, each position grows at its own gross return while wealth
grows at the portfolio gross return, so the held weights follow the self-financing
identity

.. math:: u_{t+1} = \frac{u_t \circ (1 + r_t)}{1 + u_t \cdot r_t}
   \qquad u_{t_k} = w_k

with :math:`\circ` the element-wise product, and the return of observation
:math:`t` is :math:`u_t \cdot r_t`. The weights are reset to the new targets at
:math:`t_{k+1}`, so drift accumulates over one period at a time. The identity holds
for any weights vector, including long-short and partially invested portfolios,
with an implicit cash position :math:`C_k = 1 - \sum_i w_{k,i}` that earns zero. In
closed form, with :math:`V_t` the position values and :math:`W_t` the wealth for
one unit invested at :math:`t_k`,

.. math::

   V_t = w_k \circ \prod_{s=t_k}^{t}(1+r_s),
   \qquad W_t = \sum_i V_{t,i} + C_k,
   \qquad u_t = \frac{V_{t-1}}{W_{t-1}}.

The weights are undefined once :math:`W_t \le 0`, and
:class:`~skfolio.portfolio.Portfolio` raises with the first such observation. The
window of `X` is the holding period: a direct `predict` over a long `X` drifts over
the whole window, while :func:`~skfolio.model_selection.cross_val_predict` starts a
new path at each test fold.

The drifted return decomposes exactly as

.. math:: u_t \cdot r_t = w_k \cdot r_t + d_t
   \qquad d_t = (u_t - w_k) \cdot r_t

Three quantities drive the differences between the conventions and are treated in
turn below: the mean of :math:`d_t` sets the difference between the expected
returns they estimate, its variance and its covariance with the constant-weight
return set the difference between their standard errors, and the end-of-period
holdings :math:`\tilde{w}_k` set the difference between the transaction costs they
charge.

Size of the drift
-----------------

To first order, :math:`u_{t,i} - w_{k,i} \approx w_{k,i} (C_{i,t} - C^{ptf}_t)`,
where :math:`C` denotes cumulative returns since :math:`t_k`: a held weight
deviates from its target by its asset's cumulative return relative to the
portfolio. Under serial independence this relative return is a random walk with
per-step standard deviation :math:`\sigma_d`, so the expected drift at the end of
a period is

.. math::

   \mathbb{E} \| \tilde{w}_k - w_k \|_1
   \approx \sqrt{2 / \pi} \, \sigma_d \sqrt{n}

2.3% per week for the equity example and 11% per month for the crypto example. The
magnitudes below assume a fully invested, diversified portfolio and serially
independent returns unless stated otherwise.

Expected return: bias and standard error
----------------------------------------

Two expected returns can serve as estimand before transaction costs. Each
convention is unbiased for one and biased for the other.

* The expected return of the target allocation,
  :math:`\mathbb{E}[w_k \cdot r_t]`: the return earned by holding the targets
  exactly. This is the skill of the allocation rule and the quantity the optimizer
  maximizes. Constant weights are unbiased for it, drifted weights have bias
  :math:`\mathbb{E}[d_t]`.
* The expected return of the implemented strategy,
  :math:`\mathbb{E}[u_t \cdot r_t]`: the return earned by a fund that trades to the
  targets at each rebalancing date and holds in between. Drifted weights are
  unbiased for it, constant weights have bias :math:`-\mathbb{E}[d_t]`.

**Bias.** The two estimands differ by :math:`\mathbb{E}[d_t]`. Under serial
independence the held weights are independent of :math:`r_t`, so
:math:`\mathbb{E}[d_t] = \mathbb{E}[u_t - w_k] \cdot \mathbb{E}[r_t]`: buy-and-hold
accumulates in the assets with the higher expected returns, and

.. math:: \mathbb{E}[d_t] \approx (t - t_k) \sum_i w_{k,i} (\mu_i - \mu)^2

Averaged over a period and annualized this is :math:`\bar d = (n - 1) \, s^2 / 504`:
zero when all assets share the same expected return, 0.2 bps per year for the
equity example and 16 bps for the crypto example. Under serial dependence
:math:`\mathbb{E}[d_t] \approx \rho \, \sigma_d^2` per observation, with
:math:`\rho` the lag-one autocorrelation of relative returns, positive under
momentum and negative under reversal: 13 bps per year for the equity example and
68 bps for the crypto example at :math:`\rho = 0.03`. Constant weights measure the
skill of the rule. Drifted weights measure what the fund earns, which under serial
dependence includes the return on the drift itself.

**Standard error.** The drift term changes the variance of the return series by
the relative amount

.. math::

   \gamma_\Delta
   = \frac{\operatorname{Var}(u_t \cdot r_t)
     - \operatorname{Var}(w_k \cdot r_t)}{\sigma^2}
   = \frac{\operatorname{Var}(d_t)
     + 2 \operatorname{Cov}(w_k \cdot r_t, d_t)}{\sigma^2}

and the standard error of the drifted sample mean is
:math:`\sqrt{(1 + \gamma_\Delta) \sigma^2 / T}`. The two terms are of the same
order and the second has no fixed sign. Under serial independence
:math:`\operatorname{Cov}(w_k \cdot r_t, d_t) = w_k^\top \Sigma \, \mathbb{E}[u_t - w_k]`
and, to first order, :math:`\mathbb{E}[u_{t,i} - w_{k,i}] = (t - t_k) \, w_{k,i} (\mu_i - \mu)`,
so averaged over a period

.. math::

   \frac{2 \operatorname{Cov}(w_k \cdot r_t, d_t)}{\sigma^2}
   = (n - 1) \operatorname{Cov}_w(\beta, \mu)

with :math:`\operatorname{Cov}_w` the weighted cross-sectional covariance. Drift
accumulates weight in the assets with the higher expected returns, so the drifted
series has the higher variance when those assets have the higher betas and the
lower variance when they have the lower betas. The Cauchy-Schwarz inequality bounds
the term by :math:`(n - 1) \, \sigma_\beta \, s / 252`. Adding the variance of the
drift term, :math:`\operatorname{Var}(d_t) / \sigma^2 \approx (n - 1) \sigma_d^4 / (2 N \sigma^2)`,
and the second-order term of the covariance gives the bound

.. math::

   |\gamma_\Delta| \lesssim \bar{\gamma}_\Delta
   = \frac{(n - 1) \sigma_d^4}{2 N \sigma^2}
   + (n - 1) \, \sigma_\beta \, \frac{s}{252}
   + (n - 1) \, \sigma^2 \sigma_\beta^2

0.03% for the equity example and 0.8% for the crypto example. The drifted series is
also serially dependent through the held weights, so standard errors that assume
independent observations are approximate for it, and a long-run variance estimator
is the rigorous alternative. :class:`~skfolio.model_selection.MultipleRandomizedCV`
gives a distribution over randomized paths but does not remove this dependence.

Turnover and transaction costs
------------------------------

The larger difference between the conventions comes from transaction costs. At
:math:`t_{k+1}` the optimizer moves the portfolio to :math:`w_{k+1}`. Under
constant weights the holdings before the trade are :math:`w_k`, under drifted
weights :math:`\tilde{w}_k`, and each convention measures turnover from its own
holdings:

.. math:: \text{target turnover} = \| w_{k+1} - w_k \|_1
   \qquad
   \text{executed turnover} = \| w_{k+1} - \tilde{w}_k \|_1

Target turnover measures the change in the allocation decision. Executed turnover
measures the trades a fund places, including those that bring the drifted holdings
back to the targets. Their difference is at most the drift and its sign depends on
the strategy. A momentum rule trades toward past winners, which the drift has
already overweighted, so it executes less than its target turnover. A reversal
rule trades against the drift and executes more. A rule unrelated to the drift
executes more on average, by convexity of the norm.

The one-off cost of the trade is :math:`c` times the turnover, spread over the
:math:`n` observations of the following period (Part 3). The net return of
observation :math:`t` in period :math:`k + 1` is therefore

.. math:: w_{k+1} \cdot r_t - \frac{c \, \| w_{k+1} - w_k \|_1}{n}
   \qquad \text{or} \qquad
   u_t \cdot r_t - \frac{c \, \| w_{k+1} - \tilde{w}_k \|_1}{n}

under constant and drifted weights respectively. A fund pays for executed
turnover, so the expected net return of the implemented strategy is
:math:`\mathbb{E}[u_t \cdot r_t]` minus the annual cost of executed turnover. The
constant-weight convention charges target turnover and estimates it with a bias
equal to the transaction cost gap :math:`b`. The gap is bounded by the cost rate
times the drift per rebalancing date times the number of rebalancing dates per
year:

.. math:: |b| \lesssim c \, \sqrt{2 / \pi} \, \sigma_d \sqrt{n} \times \frac{252}{n}
   = c \, \sqrt{2 / \pi} \, \sigma_d \, \frac{252}{\sqrt{n}}

12 bps per year for the equity example and 26 bps for the crypto example. The
bound scales as :math:`1 / \sqrt{n}`: less frequent rebalancing produces a larger
drift per rebalancing date but fewer of them. The realized gap reaches the bound
only when the rule trades systematically with or against the drift.

`weight_drift` switches the held weights and the turnover measure together, and
the same holdings enter `max_turnover`, so under drifted weights part of the
turnover budget goes to repairing the drift.

Mean squared error by estimand
------------------------------

**Expected return of the target allocation, before transaction costs.** The
constant-weight sample mean is unbiased with standard error
:math:`\sigma / \sqrt{T}`. The drifted sample mean has bias :math:`\bar d` and a
variance that differs by the factor :math:`1 + \gamma_\Delta`, of either sign. The
default is the estimator for this goal: it is unbiased, permutation invariant
within a period and evaluates the quantity the optimizer maximized.

**Expected net return of the implemented strategy.** The drifted estimator is
unbiased. The constant-weight estimator has bias :math:`b - \bar d`, the
transaction cost gap net of the expected drift. In annual units,

.. math:: MSE_{constant} = (b - \bar d)^2 + \frac{\sigma_{ann}^2}{T_{years}}
   \qquad
   MSE_{drifted} = (1 + \gamma_\Delta) \frac{\sigma_{ann}^2}{T_{years}}

The drifted estimator has the lower mean squared error when the squared bias
exceeds the variance it adds. Since :math:`\gamma_\Delta` is bounded by
:math:`\bar{\gamma}_\Delta`, a sufficient condition is

.. math:: |b - \bar d| > \sigma_{ann} \sqrt{\frac{\bar{\gamma}_\Delta}{T_{years}}}

5.6 bps per year for the equity example and 240 bps for the crypto example. For the
equity example the bound on the gap, 12 bps, is twice the threshold: measure
:math:`b` on the data at hand as described in the Usage section and use
`weight_drift=True` when it exceeds 5.6 bps. For the crypto example the bound on the
gap, 26 bps, and the expected drift, 16 bps, are both far below the threshold, so
the bias cannot be told apart from the variance effect, whose sign is unknown. The
two estimators are equivalent at this sample size and the default is kept for its
simpler properties.

Both the bias and the variance effect are small next to the standard error of the
mean itself, 3.4% per year for the equity example and 27% for the crypto example.
The bias matters because it is systematic: target turnover overstates the net
return of reversal strategies and understates that of momentum strategies, which
shifts decisions made against a fixed hurdle and misstates transaction cost budgets.

**Ranking.** The estimand is the difference in expected performance between
candidates. A bias common to all candidates cancels in the difference, and the
drift terms of candidates evaluated on the same history are positively correlated
and partially cancel too. Both effects favor the default for candidates of similar
style and cost level. Across styles the gap does not cancel: target turnover
overstates the costs of a momentum rule and understates those of a reversal rule,
a differential bias of up to twice the bound that does not shrink with the sample
size. A threshold of the same order applies, with the standard error of the paired
difference in place of :math:`\sigma_{ann} / \sqrt{T_{years}}`. Rank across styles
with `weight_drift=True` when the measured gap exceeds it.

**Realized path.** The estimand is the wealth path of a fund tracking the strategy
on the historical sample. Drifted weights reproduce its holdings and trades, and
with `compounded=True` its wealth path, up to the transaction cost timing residual
of Part 3. Live performance minus the drifted backtest is the implementation
shortfall [2]_. A single path is one draw of a path-dependent quantity, and
:class:`~skfolio.model_selection.MultipleRandomizedCV` turns it into a distribution
over randomized paths.

.. list-table:: The two weights conventions as estimators
   :header-rows: 1

   * - Property
     - Constant weights (default)
     - Drifted weights
   * - Return series
     - :math:`w_k \cdot r_t`
     - :math:`w_k \cdot r_t + d_t`
   * - Trades charged
     - target turnover :math:`\| w_{k+1} - w_k \|_1`
     - executed turnover :math:`\| w_{k+1} - \tilde{w}_k \|_1`
   * - Unbiased for
     - expected return of the target allocation
     - expected return of the implemented strategy, before and net of
       transaction costs
   * - Bias for the other estimand
     - :math:`b - \bar d`
     - :math:`\bar d`
   * - Standard error of the mean
     - :math:`\sigma / \sqrt{T}`
     - :math:`\sqrt{(1 + \gamma_\Delta) \, \sigma^2 / T}`, with
       :math:`|\gamma_\Delta| \lesssim \bar{\gamma}_\Delta`
   * - Choice for the net return
     - when :math:`|b - \bar d|` is below the threshold, the two estimators are
       equivalent and the default is kept, the crypto example
     - when :math:`|b - \bar d| > \sigma_{ann} \sqrt{\bar{\gamma}_\Delta / T_{years}}`,
       the equity example

Part 2: Compounding
===================

Arithmetic and geometric cumulative returns
-------------------------------------------

Compounding leaves the return series unchanged and selects how it is summarized.
The non-compounded cumulative return is :math:`\sum_t r^{ptf}_t`, with expectation
:math:`T \mu`. The compounded cumulative return is the wealth index
:math:`W_T = \prod_t (1 + r^{ptf}_t)` [3]_. Under serial independence

.. math:: \mathbb{E}[W_T] = (1 + \mu)^T
   \qquad
   \operatorname{median}[W_T] \approx e^{T g}
   \qquad
   g = \mathbb{E}[\ln(1 + r^{ptf}_t)] \approx \mu - \frac{\sigma^2}{2}

Expected wealth is governed by the arithmetic mean and does not depend on
volatility. Typical wealth, the median and the almost sure long-run growth rate,
is governed by the geometric growth rate :math:`g`, lower than :math:`\mu` by the
volatility drag :math:`\sigma^2 / 2`: 1.1% per year for the equity example and 18%
per year for the crypto example. Two consequences follow [7]_: strategies with the
same arithmetic mean have the same expected wealth at every horizon whatever their
volatilities, and a higher geometric mean at equal arithmetic mean reflects a lower
dispersion of terminal wealth, not a higher expected wealth. The two summaries
rank strategies differently when
:math:`\mu_A - \mu_B < (\sigma_A^2 - \sigma_B^2) / 2`, a condition met far more
often at crypto than at equity volatilities. Maximizing the growth rate is the
Kelly criterion [4]_, equivalent to maximizing expected logarithmic utility, and
is not optimal for other preferences [5]_. Which ranking applies depends on the
objective, expected wealth or growth rate.

Measures affected by `compounded`
---------------------------------

`compounded` affects the cumulative return series and the drawdown family: Maximum
Drawdown, Average Drawdown, Drawdown at Risk, Conditional Drawdown at Risk,
Entropic Drawdown at Risk, Ulcer Index and their ratios. Mean, variance, Sharpe
ratio, Sortino ratio, Value at Risk and Conditional Value at Risk are computed on
the per-period series and are identical under both settings. Terminal wealth is
permutation invariant like the arithmetic sum, and drawdowns are path dependent
under both settings. Compounding therefore changes the summary of a given return
series, whereas weight drift changes the return series itself. `compounded` can be
changed after construction on any :class:`~skfolio.portfolio.Portfolio` or
:class:`~skfolio.portfolio.MultiPeriodPortfolio`.

Weight drift under compounding
------------------------------

Compounding the two return series of Part 1 gives, within a period, the wealth of
a constant-mix portfolio, :math:`\prod_t (1 + w_k \cdot r_t)`, and the wealth of a
buy-and-hold portfolio,
:math:`\sum_i w_{k,i} \prod_t (1 + r_{i,t}) + (1 - \sum_i w_{k,i})`, the last term
being the implicit cash position. Constant-mix sells what rose and buys what fell
[1]_. The difference in growth rate between the two is of the same order as the
variance effect of Part 1 and its sign depends on the cross-section of expected
returns and betas in the same way. The diversification return of Booth and Fama
[6]_, :math:`g_{ptf} - \sum_i w_i g_i`, arises from imperfect correlation, is
present under both conventions, and is a different quantity from the difference
between rebalancing and buy-and-hold [7]_.

Estimation of growth rates and wealth levels
--------------------------------------------

The sample mean of :math:`\ln(1 + r^{ptf}_t)` is unbiased for :math:`g` and has,
to first order, the same standard error as the arithmetic mean, so the growth rate
is about as easy to estimate as the expected return. Wealth levels at a horizon are
not: the exponential amplifies the estimation error of the mean, and a projection
:math:`H`
years ahead carries a relative error of about
:math:`H \sigma_{ann} / \sqrt{T_{years}}`, 34% for a 10-year projection in the
equity example and 134% for a 5-year projection in the crypto example. Annualized
rates hide this, because the standard deviation of the realized annualized rate
falls as :math:`\sigma_{ann} / \sqrt{H}` while the dispersion of terminal wealth
grows with :math:`H` [8]_. Report distributions over multiple paths rather than a
single trajectory.

A mean-variance optimizer maximizes :math:`\mu - \lambda \sigma^2`, which
coincides with the growth rate :math:`\mu - \sigma^2 / 2` only at
:math:`\lambda = 1/2`. When long-run growth is the objective, set
`risk_aversion=0.5` with `objective_function=ObjectiveFunction.MAXIMIZE_UTILITY`
in :class:`~skfolio.optimization.MeanRisk`. The reporting convention does not
change the objective.

Use `compounded=False` for expectation and ranking, where the per-period measures
are identical under both settings and the arithmetic cumulative return is the
quantity whose expectation the optimizer controls. Use `compounded=True` for
realized path statistics, terminal wealth, drawdowns and reconciliation with live
performance.

.. _transaction_cost_timing:

Part 3: Transaction Cost Timing
===============================

A transaction cost is paid once per rebalancing while a position earns its return
on every observation it is held. The one-off cost :math:`c \, \| \Delta w \|_1` is
spread over the :math:`n` observations of the holding period so that it is on the
same per-period scale as the expected return in the objective, following the
:ref:`periodicity convention <periodicity_convention>`: for the equity example,
with a 10 bps one-off cost and a one-week holding period,
`transaction_costs=0.001 / 5`.

When the smoothing horizon equals the rebalancing period, the total charged equals
the one-off cost. Charging it at the rebalancing date instead changes terminal
wealth by about the trade cost times the return accrued over half the period,
:math:`c \, \| \Delta w \|_1 \, \mu \, n / 2`, below two basis points per year for
both examples, and creates spikes in the return series that distort Value at Risk,
Conditional Value at Risk and drawdown measures. `skfolio` uses smoothed
transaction costs under both weights conventions. With calendar-based rebalancing
the number of observations per period varies and the smoothed total varies with
it.

Settings by Goal
================

.. list-table:: Settings by goal
   :header-rows: 1

   * - Goal
     - Estimand
     - `weight_drift`
     - `compounded`
   * - Skill of the allocation rule, before transaction costs
     - expected return of the target allocation
     - False
     - False
   * - Expected return net of transaction costs, one strategy
     - expected net return of the implemented strategy
     - True when the measured :math:`|b - \bar d|` exceeds the threshold of
       Part 1, False otherwise
     - False
   * - Ranking strategies of similar style and cost level
     - difference in expected returns
     - False
     - False
   * - Ranking strategies across styles or cost levels
     - difference in expected net returns
     - True when the measured gap exceeds the threshold of Part 1 computed with
       the standard error of the paired difference, False otherwise
     - False
   * - Terminal wealth, drawdowns, capacity
     - statistics of the realized wealth path
     - True
     - True
   * - Reconciliation with live performance
     - realized wealth path
     - True
     - True

For the equity example the bound on the gap exceeds the threshold, so the net return
and the cross-style ranking rows resolve to `weight_drift=True` when the measured
gap does. For the crypto example the bound is below the threshold and both rows
resolve to the default. The realization rows do not depend on the example.

Usage
=====

`weight_drift` is a parameter of :class:`~skfolio.portfolio.Portfolio`.
`compounded` is a parameter of both :class:`~skfolio.portfolio.Portfolio` and
:class:`~skfolio.portfolio.MultiPeriodPortfolio`, each object summarizing its own
return series. Both default to False.

For a direct `model.predict(X)`, set `weight_drift` in the estimator's
`portfolio_params`. For :func:`~skfolio.model_selection.cross_val_predict` and
:func:`~skfolio.model_selection.online_predict`, set both conventions in the
function's `portfolio_params`: `weight_drift` is forwarded to every portfolio of the
path and the other parameters, `compounded` included, configure the returned
:class:`~skfolio.portfolio.MultiPeriodPortfolio`. A `weight_drift` given to the
function takes precedence over the estimator's value for that call.

.. code-block:: python

    from skfolio.model_selection import WalkForward, cross_val_predict
    from skfolio.optimization import MeanRisk

    model = MeanRisk(transaction_costs=0.001 / 5, max_turnover=0.3)
    pred = cross_val_predict(
        model,
        X,
        cv=WalkForward(test_size=5, train_size=252),
        portfolio_params={"weight_drift": True, "compounded": True},
        entry_rebalancing_params={"max_turnover": None},
    )

`entry_rebalancing_params` lifts `max_turnover` for the initial trade from cash. The
30% limit applies from the second rebalancing date on.

With a sequential splitter, :func:`~skfolio.model_selection.cross_val_predict` passes
each portfolio's `ending_weights` as `previous_weights` to the next fit. They equal the
targets :math:`w_k` by default and the drifted holdings :math:`\tilde{w}_k` when
`weight_drift=True`. Consequently, `transaction_costs` and `max_turnover` operate on
target turnover by default and executed turnover under drift. `weight_drift` alone is
enough to run the path sequentially. `compounded` can be changed after construction;
`weight_drift` is fixed at construction because it changes the return series itself.

A failed period contributes no returns and no holdings. The last successful
`ending_weights` are kept and passed as `previous_weights` to the next fit, so turnover
and transaction costs measured right after a failure do not reconstruct the trades of
the failed period.

Evaluate the drifted convention directly rather than reconstructing counterfactual
holdings from a default run:

.. code-block:: python

    independent_model = MeanRisk()
    cv = WalkForward(test_size=5, train_size=252)
    pred_default = cross_val_predict(independent_model, X, cv=cv)
    pred_drift = cross_val_predict(
        independent_model,
        X,
        cv=cv,
        portfolio_params={"weight_drift": True},
    )
    drifted_mean = pred_drift.annualized_mean
    convention_effect = drifted_mean - pred_default.annualized_mean

When the optimizer does not depend on `previous_weights` -- no `transaction_costs`,
no `max_turnover`, and no fallback depending on previous weights -- both runs produce
the same target sequence. Their difference in means therefore isolates the convention
effect for fixed targets. When the optimizer does depend on `previous_weights`, each
run feeds its own `ending_weights` into later optimizations, so the difference compares
two adaptive policies rather than isolating the return convention.

Transaction costs are smoothed over the holding period, management fees are charged
on the target weights, and the implicit cash position earns zero whatever the
`risk_free_rate`, under both conventions.

.. rubric:: References

.. [1] Dynamic strategies for asset allocation, Financial Analysts Journal,
   Perold and Sharpe (1988)

.. [2] The implementation shortfall: paper versus reality, Journal of Portfolio
   Management, Perold (1988)

.. [3] Quant nugget 2: linear vs. compounded returns - common pitfalls in portfolio
   management, GARP Risk Professional, Meucci (2010)

.. [4] A new interpretation of information rate, Bell System Technical Journal,
   Kelly (1956)

.. [5] The "fallacy" of maximizing the geometric mean in long sequences of investing
   or gambling, Proceedings of the National Academy of Sciences, Samuelson (1971)

.. [6] Diversification returns and asset contributions, Financial Analysts Journal,
   Booth and Fama (1992)

.. [7] The limitations of diversification return, Journal of Portfolio Management,
   Chambers and Zdanowicz (2014)

.. [8] On the risks of stocks in the long run, Financial Analysts Journal,
   Bodie (1995)

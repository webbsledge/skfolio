.. _backtesting_and_evaluation:

.. currentmodule:: skfolio

**************************
Backtesting and Evaluation
**************************

Portfolio backtests serve several distinct objectives. They can estimate allocation
skill, compare portfolio strategies, quantify implementation effects or reconstruct a
historical wealth path. Each objective may require a different construction of
portfolio returns and performance statistics.

`skfolio` is designed for the statistical evaluation of portfolio strategies: the
observed history is one realization of the process generating returns. Statistics
estimated from it can be biased and carry sampling uncertainty.

The evaluation should distinguish four objectives:

* **Allocation skill (ex-ante).** Estimate the expected return and risk of the
  target allocations selected by the optimizer, isolating the allocation decisions
  from the subsequent drift of the holdings.
* **Expected performance of the implemented strategy.** Estimate the expected
  return and risk of rebalancing to the targets and letting holdings drift between
  rebalances. Turnover and transaction costs reflect the trades needed to move from
  the drifted holdings to the new target weights.
* **Relative strategy performance.** Estimate a difference in expected performance.
  Common biases may cancel, and positively correlated sampling errors can reduce
  uncertainty in a paired comparison on the same out-of-sample dates.
* **Historical realization (ex-post).** Reconstruct the holdings, trades, wealth
  and drawdowns along the observed return path under stated implementation
  assumptions.

An accurate ex-post reconstruction does not by itself establish what to expect on a
new path. The distinction is between reconstructing an outcome and drawing statistical
conclusions from it.

These objectives guide how portfolio returns are constructed and accumulated,
and how the resulting statistics are interpreted. Two parameters govern return
construction and accumulation:

* `weight_drift` determines the weights used to compute each observation's
  return. With `False`, returns are computed at the **target weights (no drift)**,
  which remain constant within each holding period. With `True`, returns are
  computed from **drifted weights** as asset prices change between rebalancing dates.
* `compounded` determines how those returns are accumulated. With `False`,
  cumulative returns are an arithmetic sum. With `True`, they form a compounded
  wealth index. This setting also affects drawdown measures, but leaves the
  underlying return series unchanged.

Both default to `False`. The default evaluates **allocation skill (ex-ante)** at
target weights, consistent with the optimizer's linear portfolio return definition.
Cumulative returns are summarized arithmetically.

Choosing the settings
=====================

.. list-table:: Suggested starting points
   :header-rows: 1

   * - Objective
     - `weight_drift`
     - `compounded`
   * - Allocation skill (ex-ante), before transaction costs
     - `False`: target weights (no drift)
     - `False`
   * - Expected performance of the implemented strategy
     - `True`: drifted weights
     - `False`
   * - Relative strategy performance
     - `False`: target weights (no drift). `True`: drifted weights.
     - `False`
   * - Historical realization (ex-post): wealth and percentage drawdowns
     - `True`: drifted weights
     - `True`

`compounded` changes cumulative returns, drawdown measures and ratios based on
drawdowns. Measures computed from individual returns, such as mean return, variance,
VaR, CVaR, and the Sharpe and Sortino ratios, are unchanged, as are their sampling
variances.

`weight_drift` changes the underlying returns and can change both the bias and
sampling variance of performance estimates for the chosen evaluation objective.

Reconstructing holdings and trades more accurately does not necessarily improve
estimates of expected performance. An evaluation at target weights can have lower
mean squared error if its reduction in sampling variance outweighs its squared bias.
See :ref:`weight_drift_mse`.

The figure below follows two illustrative ten-year paths from the same asset return
model, with equal target weights and annual rebalancing. It adds weight drift and
then compounding while keeping the target allocations fixed. The paths were selected
to illustrate the differences. The gap is not an estimate of the typical effect of
drift or compounding.

.. include:: ../_static/backtesting/fragments/weight_drift_and_compounding.inc.rst

.. _configuring_evaluation:

Configuring an evaluation
=========================

For a direct call to an optimizer's `predict`, set the options in the optimizer's
`portfolio_params`. For :func:`~skfolio.model_selection.cross_val_predict` and
:func:`~skfolio.model_selection.online_predict`, pass them in the function's
`portfolio_params`.

The following example applies the three settings shown in the figure to the same
historical data, with rebalancing every 21 observations. `X` contains asset
returns in decimal form, ordered by date, with one column per asset.

.. code-block:: python

    from skfolio.datasets import load_sp500_dataset
    from skfolio.model_selection import WalkForward, cross_val_predict
    from skfolio.optimization import EqualWeighted
    from skfolio.preprocessing import prices_to_returns

    prices = load_sp500_dataset()
    X = prices_to_returns(prices)

    model = EqualWeighted()
    cv = WalkForward(train_size=252, test_size=21)

    target_evaluation = cross_val_predict(model, X, cv=cv)

    drifted = cross_val_predict(
        model, X, cv=cv, portfolio_params={"weight_drift": True, "compounded": False}
    )

    compounded_wealth = cross_val_predict(
        model, X, cv=cv, portfolio_params={"weight_drift": True, "compounded": True}
    )

`target_evaluation` evaluates the target weights with arithmetic accumulation.
`drifted` includes holdings drift with the same accumulation convention.
`compounded_wealth` compounds those drifted returns into wealth. The example
illustrates the settings on historical data. It does not reproduce the simulated
figure. With daily equity data, 21 observations approximate a month. Use calendar-based
splits when rebalancing must occur on specific dates.

Each walk-forward call returns a :class:`~skfolio.portfolio.MultiPeriodPortfolio`
containing one :class:`~skfolio.portfolio.Portfolio` per test period.
`weight_drift` applies to each child portfolio. `compounded` applies to both
the children and the resulting `MultiPeriodPortfolio`. Function-level values
override the optimizer's `portfolio_params`. Omitted values are inherited. See
:ref:`cross_validation` for the complete routing rules.

`compounded` can also be changed after construction. To compound the existing
`drifted` return series without repeating the evaluation, use:

.. code-block:: python

    drifted.compounded = True
    drifted.plot_cumulative_returns()

This changes the aggregate object's cumulative returns and drawdown measures. Each
child portfolio has its own `compounded` setting. `weight_drift` is fixed at
construction because it determines the return series.

Holding periods and validation
------------------------------

A direct `predict(X)` call treats all of `X` as one holding period. With drift
enabled, positions are held throughout that window. In a walk-forward evaluation,
each test period starts at its predicted target weights. Its length therefore sets
the interval between rebalances.

* :class:`~skfolio.model_selection.WalkForward` produces one sequential path.
* :func:`~skfolio.model_selection.online_predict` builds a path from a single
  stateful estimator updated with `partial_fit`. See :ref:`online_learning`.
* :class:`~skfolio.model_selection.MultipleRandomizedCV` runs walk-forward
  evaluations on randomly selected asset subsets and, optionally, contiguous time
  windows.
* :class:`~skfolio.model_selection.CombinatorialPurgedCV` evaluates combinations of
  test blocks and can train on observations later than a given test block. Its
  reconstructed paths therefore do not represent strictly forward historical
  simulations.

See :ref:`cross_validation` for purging, embargoing and supported splitters.

.. _weight_drift_evaluation:

How weight drift changes returns
================================

Let :math:`r_t` be the vector of asset returns during observation :math:`t`, and
:math:`w_k` the target weights set at rebalancing date :math:`t_k`. The holding
period runs from :math:`t_k` to just before :math:`t_{k+1}`. The equations in this
section describe returns before transaction costs and fees.

Target weights (no drift)
-------------------------

With `weight_drift=False`, each observation uses the same targets:

.. math::

   r^{target}_t = w_k \cdot r_t,
   \qquad t_k \le t < t_{k+1}.

It is consistent with the optimizer's linear portfolio definition and is the default
convention for evaluating allocation skill: the expected return and risk of the
target allocations are assessed independently of subsequent holdings drift.
Economically, it is equivalent to restoring the target weights after every
observation without charging for those within-period trades.

Drifted weights
---------------

With `weight_drift=True`, let :math:`u_t` be the weights held at the start of
observation :math:`t`. Each position grows with its asset's return:

.. math::

   r^{drifted}_t = u_t \cdot r_t,
   \qquad
   u_{t+1} = \frac{u_t \circ (1 + r_t)}{1 + u_t \cdot r_t},
   \qquad u_{t_k} = w_k,

where :math:`\circ` denotes element-wise multiplication. The weights reset to
the next targets at :math:`t_{k+1}`.

For example, start with 50% in each of two assets. If the first asset gains 10%
and the second is unchanged, the portfolio gains 5%. The next observation starts
with weights of 52.38% and 47.62%. An evaluation at target weights uses 50% and 50%
again.

If the first asset outperforms again, the drifted portfolio has a higher return
than the target-weight portfolio. If their relative performance reverses, it has
a lower return. Momentum and reversal can therefore affect the comparison.

The drift calculation assumes zero interest on cash.

`Portfolio` raises an error when wealth becomes non-positive.

Turnover and transaction costs
==============================

At the next rebalance, the strategy trades from its previous holdings to the new
target :math:`w_{k+1}`. Let :math:`\tilde w_k` denote the drifted weights at
the end of holding period :math:`k`. The two evaluations use different previous
holdings:

.. math::

   \text{target turnover} = \|w_{k+1}-w_k\|_1,
   \qquad
   \text{executed turnover} = \|w_{k+1}-\tilde w_k\|_1.

In a sequential evaluation, `weight_drift=False` uses target turnover for
transaction costs, and `weight_drift=True` uses executed turnover. Target turnover
measures changes in allocation decisions. Executed turnover also reflects price
changes since the previous trade.

Drift reduces turnover when it moves holdings toward the new targets and increases
it when it moves them away. This can reduce trading for momentum strategies and
increase it for reversal strategies.
For any given pair of targets, the triangle inequality gives

.. math::

   |\text{executed turnover} - \text{target turnover}|
   \le \|\tilde w_k-w_k\|_1.

With a sequential splitter, `cross_val_predict` passes each successful
portfolio's `ending_weights` as `previous_weights` to the next fit when
previous holdings are needed. With drift disabled, `ending_weights` contains the
target weights. With drift enabled, it contains the drifted weights at the end of
the period. Transaction costs and turnover constraints are computed from these
previous weights. Enabling drift makes the folds run sequentially and propagates
ending weights even without costs or a turnover constraint.
When fits are independent, previous weights are assigned to the predicted
portfolios afterward for turnover and cost calculations.

When asset selection changes between rebalances, turnover and transaction costs
are calculated assuming full liquidation of positions in assets absent from the
new universe.
In a pipeline, `set_output(transform="pandas")` preserves the asset names needed
to match previous holdings to the new selection. For assets absent from `X`,
`transaction_costs` must be a single rate applied to all assets or a dictionary
keyed by asset name.

For a direct `predict` call, transaction costs use the supplied `previous_weights`
under either drift setting.

The following example rebalances every five observations, applies a one-off cost
rate of 10 basis points on traded notional, and limits turnover in each asset to
30% of portfolio value at each rebalance:

.. code-block:: python

    from skfolio.optimization import MeanRisk

    holding_period = 5
    tc_rate = 0.001
    model = MeanRisk(
        transaction_costs=tc_rate / holding_period,
        max_turnover=0.3,
    )
    pred = cross_val_predict(
        model,
        X,
        cv=WalkForward(train_size=252, test_size=holding_period),
        portfolio_params={"weight_drift": True, "compounded": False},
        entry_rebalancing_params={"max_turnover": None},
    )

`max_turnover=0.3` is a per-asset constraint. Total portfolio turnover can exceed
30%. `entry_rebalancing_params` removes this constraint for the initial trade
from cash. The constraint applies from the second rebalance onward.
This limit does not apply to the assumed liquidation of positions outside the
investment universe.

With transaction costs or turnover constraints, using drifted holdings as
`previous_weights` can also change the next target allocation.
Evaluating the same sequence of targets under both settings isolates the effect
of drift on returns and turnover.

.. _transaction_cost_timing:

Transaction cost convention
---------------------------

The optimizer uses expected returns per observation. Transaction costs are paid
once at rebalancing, so a one-off cost is divided by the expected holding period
to obtain a cost per observation. Portfolio evaluation uses the same convention.
See :ref:`periodicity convention <periodicity_convention>` for the rationale and
conversion examples.

For a one-off transaction cost rate :math:`c` and an intended holding period of
:math:`n` observations, set `transaction_costs` to :math:`c/n`, as in the example above.

If turnover is :math:`q_k`, each observation in that period is reduced by
:math:`cq_k/n`. Over exactly :math:`n` observations, these deductions sum to
:math:`cq_k` in arithmetic-return units.

Charging the full cost at the trade date would produce a different return series
and, after compounding, a different wealth path. It also changes tail-risk and
drawdown measures.

Both drift settings use this cost convention. If the actual period contains
:math:`m` observations, a fixed input :math:`c/n` deducts a total of
:math:`mcq_k/n` in arithmetic-return units. This matters for calendar-based
rebalancing and incomplete periods, where the number of observations varies.

Drifted holdings and `ending_weights` are calculated before transaction costs
and management fees. These deductions are applied to portfolio returns rather than
debited from a modeled cash balance. Management fees are based on the target weights.

Return aggregation and compounding
==================================

For a given portfolio return series :math:`r^{ptf}_t`, arithmetic accumulation is

.. math::

   A_T = \sum_{t=1}^T r^{ptf}_t.

Compounding produces a wealth index, expressed per unit of starting capital:

.. math::

   W_T = \prod_{t=1}^T(1+r^{ptf}_t).

`skfolio` reports the compounded series as this wealth index. Its total return
is :math:`W_T-1`.

Estimates of compounded wealth or percentage drawdowns use `compounded=True`,
including in forward-looking evaluation. This setting does not change the
optimizer's objective.

See :ref:`data_preparation` for the discussion of simple and logarithmic returns.

For fixed targets within one holding period, before costs and fees, compounding
gives

.. math::

   W^{target} = \prod_t(1+w_k\cdot r_t),
   \qquad
   W^{drifted} = \sum_i w_{k,i}\prod_t(1+r_{i,t})
                + 1-\sum_i w_{k,i}.

The first expression describes rebalancing to the targets after every observation.
The second describes holding positions until the next scheduled rebalance.

Changing the order of asset returns within this holding period leaves both terminal
wealth values unchanged, provided wealth stays positive. At target weights, it also
leaves mean return and variance unchanged. With drifted weights, the portfolio
return series can change because holdings depend on earlier returns. Intermediate
wealth and drawdowns depend on return order in both settings.

.. _weight_drift_mse:

Estimating expected return
==========================

The sample means of the portfolio returns, computed with and without weight drift,
are two estimators of expected return. For the same target allocations, the bias
introduced by the choice of weights depends on the evaluation objective.

Before transaction costs, the return contribution from drift at observation
:math:`t` is

.. math::

   d_t = r^{drifted}_t-r^{target}_t = (u_t-w_k)\cdot r_t.

Its expected contribution to the sample mean over :math:`T` observations is
:math:`\bar d = \frac{1}{T}\sum_{t=1}^T\mathbb{E}[d_t]`.
The expected transaction cost difference per observation, :math:`b`, is the
expected cost of executed turnover minus the expected cost of target turnover.

* **Allocation skill (ex-ante).** Before transaction costs, `weight_drift=False`
  introduces no bias from drift. `weight_drift=True` introduces bias :math:`\bar d`.
* **Expected performance of the implemented strategy.** For expected net return,
  `weight_drift=True` accounts for drift and executed costs. `weight_drift=False`
  introduces bias :math:`b-\bar d`.

The expected gross-return difference between the two settings can be negligible
with frequent rebalancing. In illustrative examples, it is about **0.2 bp p.a.**
for a 50-asset equity portfolio rebalanced weekly, and about **2% p.a.** for a
four-asset crypto portfolio rebalanced quarterly.
These examples use equal target weights and geometric Brownian asset prices.
The :ref:`calculation appendix <backtesting_calculation_details>` gives the
assumptions and formulas.

The sample means also vary from one sample to another. Their sampling variances
measure this variability. In the same examples, drift increases the sampling
variance of the estimated gross mean by **less than 0.01%** for the equity
portfolio and about **3%** for the crypto portfolio. Serial correlation also
affects sampling variance. A long-run variance estimate [1]_ accounts for
dependence between observations.

Mean squared error (MSE) combines squared bias with sampling variance.
:math:`V_{\mathrm{target}}` and :math:`V_{\mathrm{drifted}}` denote the sampling
variances of the two estimated net means. When the drifted mean is unbiased for
the implemented strategy's expected net return under the return model, the MSEs
are

.. math::

   \operatorname{MSE}_{\mathrm{target}}=(b-\bar d)^2+V_{\mathrm{target}},
   \qquad
   \operatorname{MSE}_{\mathrm{drifted}}=V_{\mathrm{drifted}}.

The estimate at target weights has lower MSE when its reduction in sampling
variance exceeds its squared bias. Including drift can also reduce sampling
variance, in which case the drifted estimate has lower MSE.

For the crypto example, a portfolio volatility of 66% p.a. gives a standard error
of about **30%** for the annualized target-weight mean over five years
(:math:`66\%/\sqrt{5}`). When estimating the implemented strategy's expected gross
return, its squared bias is only about **0.45%** of its sampling variance. Its MSE
is therefore close to that of the drifted estimate. With longer samples, sampling
variance decreases while a persistent bias remains.

.. _backtesting_calculation_details:

Appendix: Calculation details
=============================

The illustrative examples assume :math:`N` equally weighted assets, rebalanced
every :math:`n` observations, with :math:`A` observations p.a. Asset prices follow
geometric Brownian motions with equal exposure to a common market factor and
independent asset-specific shocks. Returns are independent across observations,
and transaction costs are excluded.

Let :math:`\sigma_p` be the annual portfolio volatility at target weights,
:math:`\sigma_\epsilon` the annual asset-specific volatility, and :math:`s` the
standard deviation of annualized expected arithmetic asset returns.
With :math:`h=(n-1)/A`, the annualized expected gross drift contribution
:math:`\bar d_{\mathrm{ann}}` and relative change in sampling variance
:math:`\gamma` are, to leading order,

.. math::

   \bar d_{\mathrm{ann}} \approx \frac{h s^2}{2},
   \qquad
   \gamma \approx
   \frac{h(N-1)\sigma_\epsilon^4}{2N^2\sigma_p^2}.

The variance change :math:`\gamma` is measured relative to the sampling variance
of the estimated gross mean at target weights.

* **Weekly equity rebalancing.** :math:`N=50`, :math:`n=5`, :math:`A=252`,
  :math:`\sigma_p=15\%`, :math:`\sigma_\epsilon=21\%` and :math:`s=5\%`.
* **Quarterly crypto rebalancing.** :math:`N=4`, :math:`n=91`, :math:`A=365`,
  :math:`\sigma_p=66\%`, :math:`\sigma_\epsilon=87\%` and :math:`s=40\%`.

.. rubric:: References

.. [1] Newey, W. K. and West, K. D. (1987). A Simple, Positive Semi-Definite,
   Heteroskedasticity and Autocorrelation Consistent Covariance Matrix.
   *Econometrica*, 55(3), 703-708.

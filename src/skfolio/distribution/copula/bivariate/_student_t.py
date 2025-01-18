"""Bivariate Student's t Copula Estimation"""

# Copyright (c) 2025
# Author: Hugo Delatte <delatte.hugo@gmail.com>
# License: BSD 3 clause

import numpy as np
import numpy.typing as npt
import scipy.optimize as so
import scipy.special as sp
import scipy.stats as st
import sklearn.utils.validation as skv

from skfolio.distribution.copula.bivariate._base import (
    _RHO_BOUNDS,
    BaseBivariateCopula,
)
from skfolio.distribution.copula.bivariate._utils import _apply_margin_swap

# Student's t copula with dof less than 1.0 is so extremely heavy-tailed that even the
# mean (and many moments) of the distribution do not exist. So Impractical in practice,
# and dof above 50 tends to a Gaussian Copula so we limit it to the interval [1, 50] for
# improved stability and robustness.
_DOF_BOUNDS = (1.0001, 50.0)


class StudentTCopula(BaseBivariateCopula):
    r"""Bivariate Student's t Copula Estimation.

    The bivariate Student's t copula density is defined as:

    .. math::
         C_{\nu, \rho}(u, v) = T_{\nu, \rho} \Bigl(t_{\nu}^{-1}(u),\;t_{\nu}^{-1}(v)\Bigr)

    where:
    - :math:`\nu > 0` is the degrees of freedom.
    - :math:`\rho \in (-1, 1)` is the correlation coefficient.
    - :math:`T_{\nu, \rho}(x, y)` is the CDF of the bivariate \(t\)-distribution.
    - :math:`t_{\nu}^{-1}(p)` is the quantile function (inverse CDF) of the univariate
      \(t\)-distribution.

    .. note::

        Rotations are not needed for elliptical copula (e.g., Gaussian or Student-t)
        because its correlation parameter :math:`\rho \in (-1, 1)` naturally covers
        both positive and negative dependence, and they exhibit symmetric tail behavior.


    Parameters
    ----------
    use_kendall_tau_inversion : bool, default=False
        use_kendall_tau_inversion : bool, default=False
        If True, :math:`\rho` is estimated using the Kendall's tau inversion method;
        otherwise, we use the MLE (Maximum Likelihood Estimation) method (default).
        The MLE is slower but more accurate.

    kendall_tau : float, optional
        If `use_kendall_tau_inversion` is True and `kendall_tau` is provided, this
        value is used; otherwise, it is computed.

    Attributes
    ----------
    rho_ : float
        Fitted correlation coefficient (:math:`\rho`) in [-1, 1].
    dof_ : float
       Fitted degrees of freedom (:math:`\nu`) > 0.
    """

    rho_: float
    dof_: float
    _n_params = 2

    def __init__(
        self,
        use_kendall_tau_inversion: bool = False,
        kendall_tau: float | None = None,
    ):
        self.use_kendall_tau_inversion = use_kendall_tau_inversion
        self.kendall_tau = kendall_tau

    def fit(self, X: npt.ArrayLike, y=None) -> "StudentTCopula":
        """Fit the Bivariate Student's t Copula.

        If `use_kendall_tau_inversion` is True, it uses either a Kendall-based two-step
        method:

            1. Estimates the correlation parameter (:math:`\rho`) from Kendall's tau inversion.
            2. Optimizes the degrees of freedom (:math:`\nu`) by maximizing the log-likelihood.

        Otherwise, it uses the full MLE method: optimizes both :math:`\rho` and
        :math:`\nu` by maximizing the log-likelihood.

        Parameters
        ----------
        X : array-like of shape (n_observations, 2)
            An array of bivariate inputs `(u, v)` where each row represents a
            bivariate observation. Both `u` and `v` must be in the interval [0, 1],
            having been transformed to uniform marginals.

        y : None
            Ignored. Provided for compatibility with scikit-learn's API.

        Returns
        -------
        self : StudentTCopula
            Returns the instance itself.
        """
        X = self._validate_X(X, reset=True)

        if self.kendall_tau is None:
            kendall_tau = st.kendalltau(X[:, 0], X[:, 1]).statistic
        else:
            kendall_tau = self.kendall_tau

        # Either used directly or for initial guess
        rho_from_tau = np.sin((np.pi * kendall_tau) / 2.0)

        if self.use_kendall_tau_inversion:
            res = so.minimize_scalar(
                _neg_log_likelihood,
                args=(
                    rho_from_tau,
                    X,
                ),
                bounds=_DOF_BOUNDS,
                method="bounded",
            )
            if not res.success:
                raise RuntimeError(f"Optimization failed: {res.message}")
            self.dof_ = res.x
            self.rho_ = rho_from_tau
        else:
            # We'll use L-BFGS-B for the optimization because:
            # 1) The bivariate Student-t copula's negative log-likelihood is smooth,
            #    making gradient-based methods more efficient than derivative-free
            #    methods.
            # 2) L-BFGS-B directly supports simple box bounds (e.g., -1 < rho < 1,
            #    0 < nu < 50).
            # 3) It's typically faster and more stable for small-dimensional problems
            #    than more general constraint solvers (like trust-constr or SLSQP)
            result = so.minimize(
                fun=lambda x: _neg_log_likelihood(dof=x[0], rho=x[1], X=X),
                x0=np.array([3.0, rho_from_tau]),
                bounds=(_DOF_BOUNDS, _RHO_BOUNDS),
                method="L-BFGS-B",
            )
            if not result.success:
                raise RuntimeError(f"Optimization failed: {result.message}")
            self.dof_, self.rho_ = result.x

        return self

    def cdf(self, X: npt.ArrayLike) -> np.ndarray:
        """Compute the CDF of the bivariate Student-t copula.

        Parameters
        ----------
        X : array-like of shape (n_observations, 2)
            An array of bivariate inputs `(u, v)` where each row represents a
            bivariate observation. Both `u` and `v` must be in the interval `[0, 1]`,
            having been transformed to uniform marginals.

        Returns
        -------
        cdf : ndarray of shape (n_observations, )
            CDF values for each observation in X.
        """
        skv.check_is_fitted(self)
        X = self._validate_X(X, reset=False)
        cdf = st.multivariate_t.cdf(
            x=sp.stdtrit(self.dof_, X),
            loc=np.array([0, 0]),
            shape=np.array([[1, self.rho_], [self.rho_, 1]]),
            df=self.dof_,
        )
        return cdf

    def partial_derivative(
        self, X: npt.ArrayLike, first_margin: bool = False
    ) -> np.ndarray:
        r"""Compute the h-function (partial derivative) for the bivariate Student's t
        copula.

        The h-function with respect to the second margin represents the conditional
        distribution function of :math:`u` given :math:`v`:

        .. math:: \begin{aligned}
                   h(u \mid v) &= \frac{\partial C(u,v)}{\partial v} \\
                               &= t_{\nu+1}\!\left(\frac{t_\nu^{-1}(u) - \rho\,t_\nu^{-1}(v)}
                                  {\sqrt{\frac{(1-\rho^2)\left(\nu + \left(t_\nu^{-1}(v)\right)^2\right)}{\nu+1}}}\right).
                  \end{aligned}

        where:
            - :math:`\nu > 0` is the degrees of freedom.
            - :math:`\rho \in (-1, 1)` is the correlation coefficient.
            - :math:`t_{\nu}^{-1}(p)` is the quantile function (inverse CDF) of the
              univariate \(t\)-distribution.

        Parameters
        ----------
        X : array-like of shape (n_observations, 2)
            An array of bivariate inputs `(u, v)` where each row represents a
            bivariate observation. Both `u` and `v` must be in the interval `[0, 1]`,
            having been transformed to uniform marginals.

        first_margin : bool, default False
            If True, compute the partial derivative with respect to the first
            margin `u`; ,otherwise, compute the partial derivative with respect to the
            second margin `v`.

        Returns
        -------
        p : ndarray of shape (n_observations, )
            h-function values :math:`h(u \mid v) \;=\; p` for each observation in X.
        """
        skv.check_is_fitted(self)
        X = self._validate_X(X, reset=False)
        X = _apply_margin_swap(X, first_margin=first_margin)
        # Compute the inverse CDF (percent point function) using stdtrit for better
        # performance
        u_inv, v_inv = sp.stdtrit(self.dof_, X).T
        # Compute the denominator: sqrt((1 - rho^2) * (nu + y^2) / (nu + 1))
        z = (u_inv - self.rho_ * v_inv) / (
            np.sqrt((1 - self.rho_**2) * (self.dof_ + v_inv**2) / (self.dof_ + 1))
        )
        # Student's t CDF with (nu+1) degrees of freedom using stdtr for better
        # performance
        p = sp.stdtr(self.dof_ + 1, z)
        return p

    def inverse_partial_derivative(
        self, X: npt.ArrayLike, first_margin: bool = False
    ) -> np.ndarray:
        r"""Compute the inverse of the bivariate copula's partial derivative, commonly
        known as the inverse h-function [1]_.

        Let :math:`C(u, v)` be a bivariate copula. The h-function with respect to the
        second margin is defined by

        .. math::
            h(u \mid v) \;=\; \frac{\partial\,C(u, v)}{\partial\,v},

        which is the conditional distribution of :math:`U` given :math:`V = v`.
        The **inverse h-function**, denoted :math:`h^{-1}(p \mid v)`, is the unique
        value :math:`u \in [0,1]` such that

        .. math::
            h(u \mid v) \;=\; p,
            \quad \text{where } p \in [0,1].

        In practical terms, given :math:`(p, v)` in :math:`[0, 1]^2`,
        :math:`h^{-1}(p \mid v)` solves for the :math:`u` satisfying
        :math:`p = \partial C(u, v)/\partial v`.

        Parameters
        ----------
        X : array-like of shape (n_observations, 2)
            An array of bivariate inputs `(p, v)`, each in the interval `[0, 1]`.
            - The first column `p` corresponds to the value of the h-function.
            - The second column `v` is the conditioning variable.

        first_margin : bool, default False
            If True, compute the inverse partial derivative with respect to the first
            margin `u`; ,otherwise, compute the inverse partial derivative with respect
            to the second margin `v`.

        Returns
        -------
        u : ndarray of shape (n_observations, )
            A 1D-array of length `n_observations`, where each element is the computed
            :math:`u = h^{-1}(p \mid v)` for the corresponding pair in `X`.

        References
        ----------
        .. [1] "Multivariate Models and Dependence Concepts", Joe, H. (1997)
        .. [2] "An Introduction to Copulas", Nelsen, R. B. (2006)
        """
        skv.check_is_fitted(self)
        X = self._validate_X(X, reset=False)
        X = _apply_margin_swap(X, first_margin=first_margin)
        p_inv = sp.stdtrit(self.dof_ + 1, X[:, 0])
        v_inv = sp.stdtrit(self.dof_, X[:, 1])
        u_inv = (
            p_inv
            * np.sqrt((self.dof_ + v_inv**2) / (self.dof_ + 1) * (1 - self.rho_**2))
            + self.rho_ * v_inv
        )
        u = sp.stdtr(self.dof_, u_inv)
        return u

    def score_samples(self, X: npt.ArrayLike) -> np.ndarray:
        """Compute the log-likelihood of each sample (log-pdf) under the model.

        Parameters
        ----------
        X : array-like of shape (n_observations, 2)
            An array of bivariate inputs `(u, v)` where each row represents a
            bivariate observation. Both `u` and `v` must be in the interval `[0, 1]`,
            having been transformed to uniform marginals.

        Returns
        -------
        density : ndarray of shape (n_observations,)
            The log-likelihood of each sample under the fitted copula.
        """
        skv.check_is_fitted(self)
        X = self._validate_X(X, reset=False)
        log_density = _sample_scores(X=X, rho=self.rho_, dof=self.dof_)
        return log_density


def _neg_log_likelihood(dof: float, rho: float, X: np.ndarray) -> float:
    """Negative log-likelihood function for optimization.

    Parameters
    ----------
    X : array-like of shape (n_observations, 2)
        An array of bivariate inputs `(u, v)` where each row represents a
        bivariate observation. Both `u` and `v` must be in the interval `[0, 1]`,
        having been transformed to uniform marginals.

    rho : float
        Correlation copula parameter.

    dof : float
        Degree of freedom copula parameter.

    Returns
    -------
    value : float
        The negative log-likelihood value.
    """
    return -np.sum(_sample_scores(X=X, rho=rho, dof=dof))


def _sample_scores(X: np.ndarray, rho: float, dof: float) -> np.ndarray:
    """Compute the log-likelihood of each sample (log-pdf) under the bivariate
    Gaussian copula model.

    Parameters
    ----------
    X : array-like of shape (n_observations, 2)
        An array of bivariate inputs `(u, v)` where each row represents a
        bivariate observation. Both `u` and `v` must be in the interval `[0, 1]`,
        having been transformed to uniform marginals.

    rho : float
        Gaussian copula parameter.

    Returns
    -------
    density : ndarray of shape (n_observations,)
        The log-likelihood of each sample under the fitted copula.

    Raises
    ------
    ValueError
        If rho is not in (-1, 1) or dof is not positive.
    """
    if not (-1.0 <= rho <= 1.0):
        raise ValueError("rho must be between -1 and 1.")
    if not 1.0 <= dof <= 50:
        raise ValueError("Degrees of freedom `dof` must be between 1 and 50.")

    # Inverse CDF (ppf) using stdtrit for better performance
    x, y = sp.stdtrit(dof, X).T

    a = 1.0 - rho**2
    log_density = (
        sp.gammaln((dof + 2.0) / 2.0)
        + sp.gammaln(dof / 2.0)
        - 2.0 * sp.gammaln((dof + 1.0) / 2.0)
        - np.log(a) / 2
        + (dof + 1.0) / 2.0 * (np.log1p(x**2 / dof) + np.log1p(y**2 / dof))
        - (dof + 2.0) / 2.0 * np.log1p((x**2 - 2 * rho * x * y + y**2) / a / dof)
    )
    return log_density

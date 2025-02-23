"""Base Univariate Estimator"""

# Copyright (c) 2025
# Authors: The skfolio developers
# Credits: Matteo Manzi, Vincent Maladière, Carlo Nicolini
# SPDX-License-Identifier: BSD-3-Clause

import warnings
from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
import scipy.stats as st
import sklearn.base as skb
import sklearn.utils as sku
import sklearn.utils.validation as skv


class BaseUnivariateDist(skb.BaseEstimator, ABC):
    """Base Univariate Distribution Estimator.

    This abstract class serves as a foundation for univariate distribution models
    based on scipy.
    """

    _scipy_model: st.rv_continuous

    @property
    @abstractmethod
    def scipy_params(self) -> dict[str, float]:
        """Dictionary of parameters to pass to the underlying SciPy distribution"""
        pass

    @property
    @abstractmethod
    def fitted_repr(self) -> str:
        """String representation of the fitted univariate distribution"""
        pass

    @abstractmethod
    def fit(self, X: npt.ArrayLike, y=None) -> "BaseUnivariateDist":
        """Fit the univariate distribution model.

        Parameters
        ----------
        X : array-like of shape (n_observations, 1)
            The input data. X must contain a single column.


        y : None
            Ignored. Provided for compatibility with scikit-learn's API.

        Returns
        -------
        self : BaseUnivariateDist
            Returns the instance itself.
        """
        pass

    def _validate_X(self, X: npt.ArrayLike, reset: bool) -> np.ndarray:
        """Validate and convert the input data X.

        Parameters
        ----------
        X : array-like of shape (n_observations, 1)
            The input data. X must contain a single column.

        reset : bool, default=True
            Whether to reset the `n_features_in_` attribute.
            If False, the input will be checked for consistency with data
            provided when reset was last True.

        Returns
        -------
        validated_X : ndarray of shape (n_observations, 1).
            The validated input array
        """
        X = skv.validate_data(self, X, dtype=np.float64, reset=reset)
        if X.shape[1] != 1:
            raise ValueError(
                "X should should contain a single column for Univariate Distribution"
            )

        return X

    def score_samples(self, X: npt.ArrayLike) -> np.ndarray:
        """Compute the log-likelihood of each sample (log-pdf) under the model.

        Parameters
        ----------
        X : array-like of shape (n_observations, 1)
            An array of points at which to evaluate the log-probability density.
            The data should be a single feature column.

        Returns
        -------
        density : ndarray of shape (n_observations,)
            Log-likelihood values for each observation in X.
        """
        X = self._validate_X(X, reset=False)
        log_density = self._scipy_model.logpdf(X, **self.scipy_params)
        return log_density

    def score(self, X: npt.ArrayLike, y=None):
        """Compute the total log-likelihood under the model.

        Parameters
        ----------
        X : array-like of shape (n_observations, 1)
            An array of data points for which the total log-likelihood is computed.

        y : None
            Ignored. Provided for compatibility with scikit-learn's API.

        Returns
        -------
        logprob : float
            The total log-likelihood (sum of log-pdf values).
        """
        return np.sum(self.score_samples(X))

    def sample(self, n_samples: int = 1, random_state: int | None = None):
        """Generate random samples from the fitted distribution.

        Currently, this is implemented only for gaussian and tophat kernels.

        Parameters
        ----------
        n_samples : int, default=1
            Number of samples to generate.

        random_state : int, RandomState instance or None, default=None
            Seed or random state to ensure reproducibility.

        Returns
        -------
        X : array-like of shape (n_samples, 1)
            List of samples.
        """
        skv.check_is_fitted(self)
        rng = sku.check_random_state(random_state)
        sample = self._scipy_model.rvs(
            size=(n_samples, 1), random_state=rng, **self.scipy_params
        )
        return sample

    def aic(self, X: npt.ArrayLike) -> float:
        r"""Compute the Akaike Information Criterion (AIC) for the model given data X.

        The AIC is defined as:

        .. math::
            \mathrm{AIC} = -2 \, \log L \;+\; 2 k,

        where

        - :math:`\log L` is the (maximized) total log-likelihood
        - :math:`k` is the number of parameters in the model

        A lower AIC value indicates a better trade-off between model fit and complexity.

        Parameters
        ----------
        X : array-like of shape (n_observations, 1)
            The input data on which to compute the AIC.


        Notes
        -----
        In practice, both AIC and BIC measure the trade-off between model fit and
        complexity, but BIC tends to prefer simpler models for large :math:`n`
        because of the :math:`\ln(n)` term.

        Returns
        -------
        aic : float
            The AIC of the fitted model on the given data.

        References
        ----------
        .. [1] "A new look at the statistical model identification", Akaike (1974).
        """
        log_likelihood = self.score(X)
        k = len(self.scipy_params)
        return 2 * (k - log_likelihood)

    def bic(self, X: npt.ArrayLike) -> float:
        r"""Compute the Bayesian Information Criterion (BIC) for the model given data X.

        The BIC is defined as:

        .. math::
           \mathrm{BIC} = -2 \, \log L \;+\; k \,\ln(n),

        where

        - :math:`\log L` is the (maximized) total log-likelihood
        - :math:`k` is the number of parameters in the model
        - :math:`n` is the number of observations

        A lower BIC value suggests a better fit while imposing a stronger penalty
        for model complexity than the AIC.

        Parameters
        ----------
        X : array-like of shape (n_observations, 1)
            The input data on which to compute the BIC.

        Returns
        -------
        bic : float
           The BIC of the fitted model on the given data.

        Notes
        -----
        In practice, both AIC and BIC measure the trade-off between model fit and
        complexity, but BIC tends to prefer simpler models for large :math:`n`
        because of the :math:`\ln(n)` term.

        References
        ----------
        .. [1]  "Estimating the dimension of a model", Schwarz, G. (1978).
        """
        log_likelihood = self.score(X)
        n = X.shape[0]
        k = len(self.scipy_params)
        return -2 * log_likelihood + k * np.log(n)

    def cdf(self, X: npt.ArrayLike) -> np.ndarray:
        """Compute the cumulative distribution function (CDF) for the given data.

        Parameters
        ----------
        X : array-like of shape (n_observations, 1)
            Data points at which to evaluate the CDF.

        Returns
        -------
        cdf : ndarray of shape (n_observations, 1)
            The CDF evaluated at each data point.
        """
        skv.check_is_fitted(self)
        return self._scipy_model.cdf(X, **self.scipy_params)

    def ppf(self, X: npt.ArrayLike) -> np.ndarray:
        """Compute the percent point function (inverse of the CDF) for the given
         probabilities.

        Parameters
        ----------
        X : array-like of shape (n_observations, 1)
            Probabilities for which to compute the corresponding quantiles.

        Returns
        -------
         ppf : ndarray of shape (n_observations, 1)
            The quantiles corresponding to the given probabilities.
        """
        skv.check_is_fitted(self)
        return self._scipy_model.ppf(X, **self.scipy_params)

    def plot_pdf(self, title: str | None = None) -> go.Figure:
        """Plot the probability density function (PDF).

        Parameters
        ----------
        title : str, optional
           The title for the plot. If not provided, a default title based on the fitted
           model's representation is used.

        Returns
        -------
        fig : go.Figure
           A Plotly figure object containing the PDF plot.
        """
        skv.check_is_fitted(self)

        if title is None:
            title = f"PDF of {self.fitted_repr}"

        # Compute the quantile-based range
        lower_bound = self.ppf(1e-3)
        upper_bound = self.ppf(1 - 1e-3)

        # Generate x values across this range
        x = np.linspace(lower_bound, upper_bound, 1000)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            pdfs = np.exp(self.score_samples(x.reshape(-1, 1)))

        fig = go.Figure(
            go.Scatter(
                x=x,
                y=pdfs.flatten(),
                mode="lines",
                fill="tozeroy",
                name="Student T PDF",
            )
        )
        fig.update_layout(
            title=title,
            xaxis_title="x",
            yaxis_title="Probability Density",
        )
        fig.update_xaxes(
            tickformat=".0%",
        )
        return fig

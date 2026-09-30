"""Small statistics helpers: scaling fitted on training data, and prediction scores."""
from __future__ import annotations

import numpy as np
from scipy import stats as _st


# --------------------------------------------------------------------------- scaling
class Standardiser:
    """z-scores columns with statistics from the data it is fitted on (NaNs ignored).

    Fit on training data only and apply to test data, so that test patients never
    influence the scaling. Constant columns are left centred and unscaled."""

    def __init__(self, ddof: int = 1):
        self.ddof = ddof

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = np.nanmean(X, axis=0)
        sd = np.nanstd(X, axis=0, ddof=self.ddof)
        self.scale_ = np.where((sd > 0) & np.isfinite(sd), sd, 1.0)
        return self

    def transform(self, X):
        return (np.asarray(X, dtype=np.float64) - self.mean_) / self.scale_

    def inverse_transform(self, Z):
        return np.asarray(Z) * self.scale_ + self.mean_

    def fit_transform(self, X):
        return self.fit(X).transform(X)



# --------------------------------------------------------------------------- scores
def crps_gaussian(observed, mean, sd) -> np.ndarray:
    """Continuous ranked probability score of N(mean, sd^2) for the observed values (lower is better).

    CRPS = sd * [z (2 Phi(z) - 1) + 2 phi(z) - 1/sqrt(pi)],  z = (observed - mean) / sd.
    sd <= 0 gives the absolute error (the limit of a point forecast)."""
    observed, mean, sd = np.broadcast_arrays(*(np.atleast_1d(np.asarray(a, dtype=np.float64))
                                               for a in (observed, mean, sd)))
    out = np.array(np.abs(observed - mean), dtype=np.float64)
    ok = sd > 0
    z = (observed[ok] - mean[ok]) / sd[ok]
    out[ok] = sd[ok] * (z * (2 * _st.norm.cdf(z) - 1) + 2 * _st.norm.pdf(z) - 1 / np.sqrt(np.pi))
    return out


def gaussian_nll(observed, mean, var, eps: float = 1e-6) -> np.ndarray:
    """Per-observation Gaussian negative log likelihood, 0.5 [log(2 pi var) + (y - mean)^2 / var].

    The original (gaussian_negative_log_likelihood_loss.m) dropped log(2 pi) and added a constant 1."""
    var = np.maximum(np.asarray(var, dtype=np.float64), eps)
    r = np.asarray(observed, dtype=np.float64) - np.asarray(mean, dtype=np.float64)
    return 0.5 * (np.log(2 * np.pi * var) + r ** 2 / var)


def regression_scores(y, pred) -> dict:
    """r, r^2 (squared correlation, what the original called R2), out-of-sample R^2, MAE, RMSE.

    The original reported `regress(y, [1 pred])` R^2, which is the squared correlation:
    it rewards predictions that are correlated with the outcome even when they are
    badly scaled or offset. `R2` here is 1 - SSE/SST, which penalises that too."""
    y = np.asarray(y, dtype=np.float64).ravel()
    pred = np.asarray(pred, dtype=np.float64).ravel()
    ok = np.isfinite(y) & np.isfinite(pred)
    y, pred = y[ok], pred[ok]
    if len(y) < 3:
        return dict(n=len(y), r=np.nan, r2=np.nan, R2=np.nan, mae=np.nan, rmse=np.nan)
    r = np.corrcoef(y, pred)[0, 1] if np.std(pred) > 0 and np.std(y) > 0 else np.nan
    sst = ((y - y.mean()) ** 2).sum()
    return dict(n=int(len(y)), r=float(r), r2=float(r ** 2) if np.isfinite(r) else np.nan,
                R2=float(1 - ((y - pred) ** 2).sum() / sst) if sst > 0 else np.nan,
                mae=float(np.abs(y - pred).mean()), rmse=float(np.sqrt(((y - pred) ** 2).mean())))


# --------------------------------------------------------------------------- correlation and multiple comparisons

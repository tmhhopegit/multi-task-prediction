"""Predicting behavioural change from brain change (RunPredictivePLS.m), as models that can be
cross-validated like any other.

    PCAModel         principal components of the (many) brain features, keeping those that explain
                     at least `min_explained` of the variance, then any model on the component scores.
                     This is what RunPredictivePLS.m's code did (with a multi-task GP), except that
                     the PCA is fitted on the training patients only: the original computed it once
                     on all patients before the leave-one-out loop.
    PeakVoxelModel   what RunPredictivePLS.m's header comment described: on the training patients,
                     correlate every voxel with the outcome, take the peak voxel, fit a regression on
                     it, and predict the held-out patient. One peak per output.

direction_accuracy() reports how often improvers and decliners are predicted to improve / decline
(the ImpAcc / DecAcc of the original's commented-out summary).

(The original's `ncomps` argument was unused, and its function name promised PLS.)
"""
from __future__ import annotations

import numpy as np
from scipy import stats as _st

from .models import make_model


class PCAModel:
    def __init__(self, base: str = "mtgp", min_explained: float = 0.01, max_components: int | None = None,
                 seed: int = 0, **base_args):
        self.base, self.min_explained, self.max_components = base, min_explained, max_components
        self.seed, self.base_args = seed, base_args

    def _scores(self, X):
        return (np.asarray(X, dtype=np.float64) - self.mean_) @ self.components_.T

    def fit(self, X, Y):
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = X.mean(0)
        U, s, Vt = np.linalg.svd(X - self.mean_, full_matrices=False)
        explained = s ** 2 / (s ** 2).sum()
        keep = np.flatnonzero(explained >= self.min_explained)
        if self.max_components:
            keep = keep[:self.max_components]
        if len(keep) == 0:
            keep = np.array([0])
        self.components_ = Vt[keep]
        self.explained_ = explained[keep]
        self.model_ = make_model(self.base, seed=self.seed, **self.base_args).fit(self._scores(X), Y)
        return self

    def predict(self, X, return_std: bool = False):
        return self.model_.predict(self._scores(X), return_std=return_std) if return_std \
            else self.model_.predict(self._scores(X))

    def predict_given(self, X, Y_partial, return_std: bool = False):
        return self.model_.predict_given(self._scores(X), Y_partial, return_std=return_std)


class PeakVoxelModel:
    """sign: +1 = only positive correlations, -1 = only negative, 0 = either (the original's `mode`)."""

    def __init__(self, sign: int = 0):
        self.sign = sign

    def fit(self, X, Y):
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        Y = Y[:, None] if Y.ndim == 1 else Y
        self.peaks_, self.coef_ = [], []
        for j in range(Y.shape[1]):
            ok = ~np.isnan(Y[:, j])
            x, y = X[ok], Y[ok, j]
            xc, yc = x - x.mean(0), y - y.mean()
            den = np.sqrt((xc ** 2).sum(0) * (yc ** 2).sum())
            r = np.divide(xc.T @ yc, den, out=np.zeros(x.shape[1]), where=den > 0)
            if self.sign > 0:
                r = np.where(r > 0, r, -np.inf)
            elif self.sign < 0:
                r = np.where(r < 0, -r, -np.inf)
            else:
                r = np.abs(r)
            v = int(np.argmax(r))
            b = np.polyfit(x[:, v], y, 1)
            self.peaks_.append(v)
            self.coef_.append(b)
        return self

    def predict(self, X, return_std: bool = False):
        X = np.asarray(X, dtype=np.float64)
        return np.column_stack([np.polyval(b, X[:, v]) for v, b in zip(self.peaks_, self.coef_)])

    def predict_given(self, X, Y_partial, return_std: bool = False):
        return self.predict(X)


def direction_accuracy(change, predicted) -> dict:
    """Share of improvers (change > 0) predicted > 0, and of decliners (change < 0) predicted < 0, plus a
    t-test of the predictions of improvers vs the rest (the original's ImpClass)."""
    change = np.asarray(change, dtype=np.float64).ravel()
    predicted = np.asarray(predicted, dtype=np.float64).ravel()
    ok = np.isfinite(change) & np.isfinite(predicted)
    c, p = change[ok], predicted[ok]
    imp, dec = c > 0, c < 0
    t = _st.ttest_ind(p[imp], p[~imp]) if imp.any() and (~imp).any() else None
    return {"improvers_correct": float((p[imp] > 0).mean()) if imp.any() else np.nan,
            "decliners_correct": float((p[dec] < 0).mean()) if dec.any() else np.nan,
            "improver_t": float(t.statistic) if t else np.nan, "improver_p": float(t.pvalue) if t else np.nan}

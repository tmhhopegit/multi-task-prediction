"""Linear single-output models and their multi-output counterparts.

    single output (one model per output)       multi-output (one joint model)
    IndependentRidge   (ridge per output)      ReducedRankRidge   (ridge, then the fitted values
                                                                   projected onto a few shared directions)
    IndependentLasso   (lasso per output)      MultiTaskLasso     (L2,1 penalty: the same features
                                                                   chosen for every output, as MALSAR's
                                                                   Least_L21, whose INSTALL.m was here)
    PLS1               (PLS per output)        PLS2               (one PLS model for all outputs)

Penalties and numbers of components are chosen by inner cross-validation on the training data.
Inputs are z-scored with training statistics.

Missing outputs: the single-output models use each output's own complete rows. The joint models
need complete rows, so they drop any row with a missing output (a real disadvantage of these
models, which the multi-task GP does not share).

AugmentedSingle is a stronger single-output baseline for predicting a patient's missing outputs
from the ones they have: each output's model also takes the other outputs as inputs (missing ones
filled in by predictions from the inputs alone).
"""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LassoCV, MultiTaskLassoCV, RidgeCV
from sklearn.model_selection import KFold

from .stats import Standardiser

ALPHAS = np.logspace(-3, 3, 25)


def _prep(X, Y):
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    return X, (Y[:, None] if Y.ndim == 1 else Y)


class _Base:
    def predict_given(self, X, Y_partial, return_std: bool = False):
        """These models cannot use a patient's other known outputs, so this is predict()."""
        return self.predict(X)

    def _scale(self, X):
        return self.xs_.transform(X)


class IndependentRidge(_Base):
    def fit(self, X, Y):
        X, Y = _prep(X, Y)
        self.xs_ = Standardiser().fit(X)
        Z = self._scale(X)
        self.models_ = [RidgeCV(alphas=ALPHAS).fit(Z[~np.isnan(Y[:, j])], Y[~np.isnan(Y[:, j]), j])
                        for j in range(Y.shape[1])]
        return self

    def predict(self, X, return_std=False):
        Z = self._scale(np.asarray(X, dtype=np.float64))
        return np.column_stack([m.predict(Z) for m in self.models_])


class ReducedRankRidge(_Base):
    """Ridge on all outputs, then fitted values projected onto their top `rank` principal directions
    (Izenman's reduced-rank regression with a ridge penalty). rank=None: chosen by 5-fold CV."""

    def __init__(self, rank: int | None = None, seed: int = 0):
        self.rank, self.seed = rank, seed

    @staticmethod
    def _fit_rrr(Z, Y, alpha, rank):
        n, p = Z.shape
        B = np.linalg.solve(Z.T @ Z + alpha * np.eye(p), Z.T @ Y)
        _, _, Vt = np.linalg.svd(Z @ B, full_matrices=False)
        V = Vt[:rank].T
        return B @ V @ V.T

    def fit(self, X, Y):
        X, Y = _prep(X, Y)
        ok = ~np.isnan(Y).any(1)
        X, Y = X[ok], Y[ok]
        self.xs_ = Standardiser().fit(X)
        self.ys_ = Standardiser().fit(Y)
        Z, Yz = self._scale(X), self.ys_.transform(Y)
        M = Y.shape[1]
        ranks = [self.rank] if self.rank is not None else list(range(1, M + 1))
        best = (np.inf, None, None)
        folds = list(KFold(min(5, len(Z)), shuffle=True, random_state=self.seed).split(Z))
        for a in ALPHAS[::3]:
            for r in ranks:
                err = sum(((Yz[te] - Z[te] @ self._fit_rrr(Z[tr], Yz[tr], a, r)) ** 2).sum() for tr, te in folds)
                if err < best[0]:
                    best = (err, a, r)
        self.alpha_, self.rank_ = best[1], best[2]
        self.coef_ = self._fit_rrr(Z, Yz, self.alpha_, self.rank_)
        return self

    def predict(self, X, return_std=False):
        return self.ys_.inverse_transform(self._scale(np.asarray(X, dtype=np.float64)) @ self.coef_)


class IndependentLasso(_Base):
    def fit(self, X, Y):
        X, Y = _prep(X, Y)
        self.xs_ = Standardiser().fit(X)
        Z = self._scale(X)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            self.models_ = [LassoCV(cv=5, random_state=0).fit(Z[~np.isnan(Y[:, j])], Y[~np.isnan(Y[:, j]), j])
                            for j in range(Y.shape[1])]
        return self

    def predict(self, X, return_std=False):
        Z = self._scale(np.asarray(X, dtype=np.float64))
        return np.column_stack([m.predict(Z) for m in self.models_])

    def selected(self) -> np.ndarray:
        return np.column_stack([m.coef_ != 0 for m in self.models_])


class MultiTaskLasso(_Base):
    def fit(self, X, Y):
        X, Y = _prep(X, Y)
        ok = ~np.isnan(Y).any(1)
        self.xs_ = Standardiser().fit(X[ok])
        self.ys_ = Standardiser().fit(Y[ok])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            self.model_ = MultiTaskLassoCV(cv=5, random_state=0).fit(self._scale(X[ok]), self.ys_.transform(Y[ok]))
        return self

    def predict(self, X, return_std=False):
        return self.ys_.inverse_transform(self.model_.predict(self._scale(np.asarray(X, dtype=np.float64))))

    def selected(self) -> np.ndarray:
        return (self.model_.coef_ != 0).T


def _pls_cv(Z, Y, max_comp, seed):
    folds = list(KFold(min(5, len(Z)), shuffle=True, random_state=seed).split(Z))
    errs = []
    for c in range(0, max_comp + 1):
        e = 0.0
        for tr, te in folds:
            if c == 0:
                pred = np.tile(Y[tr].mean(0), (len(te), 1))
            else:
                pred = PLSRegression(n_components=min(c, len(tr) - 1), scale=False).fit(Z[tr], Y[tr]).predict(Z[te])
            e += ((np.asarray(pred).reshape(len(te), -1) - Y[te]) ** 2).sum()
        errs.append(e)
    return int(np.argmin(errs))


class PLS1(_Base):
    def __init__(self, max_components: int = 10, seed: int = 0):
        self.max_components, self.seed = max_components, seed

    def fit(self, X, Y):
        X, Y = _prep(X, Y)
        self.xs_ = Standardiser().fit(X)
        Z = self._scale(X)
        self.models_ = []
        for j in range(Y.shape[1]):
            ok = ~np.isnan(Y[:, j])
            y = Y[ok, j][:, None]
            c = _pls_cv(Z[ok], y, min(self.max_components, Z.shape[1], ok.sum() - 2), self.seed)
            self.models_.append((c, y.mean(), PLSRegression(n_components=c, scale=False).fit(Z[ok], y) if c else None))
        return self

    def predict(self, X, return_std=False):
        Z = self._scale(np.asarray(X, dtype=np.float64))
        return np.column_stack([np.full(len(Z), mu) if m is None else np.asarray(m.predict(Z)).ravel()
                                for c, mu, m in self.models_])


class PLS2(_Base):
    def __init__(self, max_components: int = 10, seed: int = 0):
        self.max_components, self.seed = max_components, seed

    def fit(self, X, Y):
        X, Y = _prep(X, Y)
        ok = ~np.isnan(Y).any(1)
        self.xs_ = Standardiser().fit(X[ok])
        self.ys_ = Standardiser().fit(Y[ok])
        Z, Yz = self._scale(X[ok]), self.ys_.transform(Y[ok])
        self.n_components_ = _pls_cv(Z, Yz, min(self.max_components, Z.shape[1], len(Z) - 2), self.seed)
        self.model_ = PLSRegression(n_components=self.n_components_, scale=False).fit(Z, Yz) \
            if self.n_components_ else None
        return self

    def predict(self, X, return_std=False):
        Z = self._scale(np.asarray(X, dtype=np.float64))
        if self.model_ is None:
            return self.ys_.inverse_transform(np.zeros((len(Z), len(self.ys_.mean_))))
        return self.ys_.inverse_transform(np.asarray(self.model_.predict(Z)).reshape(len(Z), -1))


class AugmentedSingle(_Base):
    """Single-output models that also take the *other outputs* as inputs (for predict_given).

    make_base: a function returning a fresh single-output-capable model (fit(X, Y) / predict(X)).
    For predict(X) alone, the base models are used. For predict_given, output j's model gets
    [X, other outputs], with any missing other outputs filled in by the base predictions."""

    def __init__(self, make_base):
        self.make_base = make_base

    def fit(self, X, Y):
        X, Y = _prep(X, Y)
        self.base_ = self.make_base().fit(X, Y)
        filled = np.where(np.isnan(Y), self.base_.predict(X), Y)
        self.aug_ = []
        for j in range(Y.shape[1]):
            ok = ~np.isnan(Y[:, j])
            others = np.delete(filled, j, axis=1)
            self.aug_.append(self.make_base().fit(np.c_[X, others][ok], Y[ok, j]))
        return self

    def predict(self, X, return_std=False):
        return self.base_.predict(X) if not return_std else self.base_.predict(X, return_std=True)

    def predict_given(self, X, Y_partial, return_std: bool = False):
        X = np.asarray(X, dtype=np.float64)
        Yp = np.asarray(Y_partial, dtype=np.float64)
        filled = np.where(np.isnan(Yp), self.base_.predict(X), Yp)
        means, sds = [], []
        for j, m in enumerate(self.aug_):
            o = m.predict(np.c_[X, np.delete(filled, j, axis=1)], return_std=True)
            mu, sd = o if isinstance(o, tuple) else (o, None)
            means.append(np.asarray(mu).reshape(len(X), -1)[:, 0])
            sds.append(None if sd is None else np.asarray(sd).reshape(len(X), -1)[:, 0])
        mean = np.column_stack(means)
        if return_std and all(s is not None for s in sds):
            return mean, np.column_stack(sds)
        return mean

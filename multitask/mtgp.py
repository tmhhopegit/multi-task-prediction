"""Multi-output Gaussian process: the linear model of coregionalisation (LMC).

    cov(f_t(x), f_s(x')) = sum_q  B_q[t, s] * k_q(x, x'),     B_q = L_q L_q^T  (L_q: outputs x rank)
    y_t(x) = f_t(x) + noise_t

Each k_q is a squared-exponential kernel with its own ARD length scales, and B_q says how
strongly the outputs share that latent function. With n_latent = 1 this is the "intrinsic
coregionalisation model" of Bonilla, Chai & Williams (2008), the model the MTGP toolbox fitted
(MGP_Bonilla.m, TrainMultiGP_Fast*.m). With n_latent > 1, outputs can depend on shared processes
with different smoothness, which is what the convolution-process kernels of the multigp toolbox
(TrainMultiGP, TrainMultiGP_Sparse: 'gg' kernels with latent forces) were used for.

Missing outputs (NaN in Y) need no special handling: the model is fitted on the observed
(row, output) pairs only.

predict_given(X, Y_partial) predicts a new patient's missing outputs *given* the outputs they
do have, by conditioning the fitted GP on those extra observations (no refitting). This is where
a multi-output model can beat independent models, which cannot use that information.

Hyperparameters are fitted by maximising the log marginal likelihood with analytic gradients
(L-BFGS, several starts). prior_var adds a N(0, prior_var) prior on every log-hyperparameter and
on the entries of L_q; the default is no prior, because a prior on the length scales penalises a
single-output GP (less data to overcome it) more than a multi-output one and so biases the
comparison (see README). Inputs and each output are z-scored with the training data's statistics.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

from .stats import Standardiser


def _kx(A, B, w, same=False):
    Aw, Bw = A * np.sqrt(w), B * np.sqrt(w)
    d2 = np.maximum((Aw ** 2).sum(1)[:, None] + (Bw ** 2).sum(1)[None, :] - 2 * Aw @ Bw.T, 0)
    if same:
        np.fill_diagonal(d2, 0)
    return np.exp(-d2 / 2)


class MultiTaskGP:
    def __init__(self, n_latent: int = 1, rank: int | None = None, correlated_noise: bool = False,
                 prior_var: float | None = None, n_restarts: int = 2, max_iter: int = 300, seed: int | None = 0):
        """correlated_noise: the noise of one row's outputs is correlated (free-form covariance S = N N^T)
        instead of independent. This lets the model represent variation that the outputs of one patient
        share but the inputs do not explain (e.g. general severity), which is what makes a patient's
        known outputs informative about their missing ones."""
        self.n_latent, self.rank, self.prior_var = n_latent, rank, prior_var
        self.correlated_noise = correlated_noise
        self.n_restarts, self.max_iter, self.seed = n_restarts, max_iter, seed

    # ------------------------------------------------------------------ parameters
    def _sizes(self):
        M, r, D, Q = self.n_tasks_, self.rank_, self.n_features_, self.n_latent
        return M, r, D, Q

    def _unpack(self, theta):
        M, r, D, Q = self._sizes()
        Ls, ws, i = [], [], 0
        for _ in range(Q):
            Ls.append(theta[i:i + M * r].reshape(M, r))
            i += M * r
            ws.append(np.exp(theta[i:i + D]))
            i += D
        if self.correlated_noise:
            Nm = theta[i:i + M * M].reshape(M, M)
            S = Nm @ Nm.T + 1e-6 * np.eye(M)
        else:
            Nm = None
            S = np.diag(np.exp(theta[i:i + M]))
        return Ls, ws, (S, Nm)

    def _initial(self, rng, restart):
        M, r, D, Q = self._sizes()
        parts = []
        for q in range(Q):
            L0 = (np.eye(M, r) * 0.9 + 0.1) / np.sqrt(Q)
            logw = np.full(D, -np.log(D) + (q - (Q - 1) / 2) * 1.5)       # different smoothness per latent
            if restart:
                L0 = L0 + rng.normal(scale=0.2, size=L0.shape)
                logw = logw + rng.normal(scale=0.5, size=D)
            parts += [L0.ravel(), logw]
        if self.correlated_noise:
            N0 = np.eye(M) * np.sqrt(np.exp(-2.0)) + (rng.normal(scale=0.05, size=(M, M)) if restart else 0)
            parts.append(N0.ravel())
        else:
            parts.append(np.full(M, -2.0))
        return np.concatenate(parts)

    def _bounds(self):
        """Wide but finite: L entries within +-50, log length scales and log noise within +-15."""
        M, r, D, Q = self._sizes()
        noise = [(-50.0, 50.0)] * (M * M) if self.correlated_noise else [(-15.0, 15.0)] * M
        return ([(-50.0, 50.0)] * (M * r) + [(-15.0, 15.0)] * D) * Q + noise

    # ------------------------------------------------------------------ likelihood
    def _nlml(self, theta):
        X, t, y = self.Xo_, self.to_, self.yo_
        n = len(y)
        M, r, D, Q = self._sizes()
        Ls, ws, (S, Nm) = self._unpack(theta)
        E = self.same_row_
        Kxs, Kfts = [], []
        K = S[np.ix_(t, t)] * E
        for L, w in zip(Ls, ws):
            Kx = _kx(X, X, w, same=True)
            Kft = (L @ L.T)[np.ix_(t, t)]
            K += Kft * Kx
            Kxs.append(Kx)
            Kfts.append(Kft)
        if not np.isfinite(K).all():
            return 1e25, np.zeros_like(theta)
        try:
            C = cho_factor(K + 1e-8 * np.eye(n), lower=True)
        except (np.linalg.LinAlgError, ValueError):
            return 1e25, np.zeros_like(theta)
        alpha = cho_solve(C, y)
        nll = 0.5 * y @ alpha + np.log(np.diag(C[0])).sum() + 0.5 * n * np.log(2 * np.pi)
        W = np.outer(alpha, alpha) - cho_solve(C, np.eye(n))
        T = np.zeros((n, M))
        T[np.arange(n), t] = 1
        g = []
        for L, w, Kx, Kft in zip(Ls, ws, Kxs, Kfts):
            g.append((-(T.T @ (W * Kx) @ T @ L)).ravel())
            WK = W * Kft * Kx
            gw = np.empty(D)
            for d in range(D):
                diff2 = (X[:, d][:, None] - X[:, d][None, :]) ** 2
                gw[d] = 0.25 * w[d] * (WK * diff2).sum()
            g.append(gw)
        if self.correlated_noise:
            g.append((-(T.T @ (W * E) @ T @ Nm)).ravel())
        else:
            g.append(-0.5 * np.bincount(t, weights=np.diag(W), minlength=M) * np.diag(S))
        g = np.concatenate(g)
        if self.prior_var is not None:
            nll += 0.5 * (theta ** 2).sum() / self.prior_var
            g = g + theta / self.prior_var
        return nll, g

    # ------------------------------------------------------------------ fit
    def fit(self, X, Y):
        """X: (rows, features). Y: (rows, outputs), NaN where an output is missing."""
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        Y = Y[:, None] if Y.ndim == 1 else Y
        if np.isnan(X).any():
            raise ValueError("inputs must not be missing (outputs may be)")
        self.n_features_, self.n_tasks_ = X.shape[1], Y.shape[1]
        self.rank_ = self.n_tasks_ if self.rank is None else int(self.rank)
        self.xs_ = Standardiser().fit(X)
        self.ys_ = Standardiser().fit(Y)
        rows, tasks = np.nonzero(~np.isnan(Y))
        Xz, Yz = self.xs_.transform(X), self.ys_.transform(Y)
        self.Xo_, self.to_, self.yo_, self.ro_ = Xz[rows], tasks, Yz[rows, tasks], rows
        self.same_row_ = (rows[:, None] == rows[None, :]).astype(np.float64)
        self.n_train_ = len(X)
        rng = np.random.default_rng(self.seed)
        best = None
        for k in range(max(1, self.n_restarts)):
            theta0 = self._initial(rng, k > 0)
            res = minimize(self._nlml, theta0, jac=True, method="L-BFGS-B", bounds=self._bounds(),
                           options={"maxiter": self.max_iter})
            if best is None or res.fun < best.fun:
                best = res
        self.theta_, self.nlml_ = best.x, float(best.fun)
        self.Ls_, self.ws_, (self.S_, _) = self._unpack(best.x)
        self.n2_ = np.diag(self.S_).copy()
        self.Bs_ = [L @ L.T for L in self.Ls_]
        self._condition(self.Xo_, self.to_, self.yo_, self.ro_)
        return self

    def _cov(self, XA, tA, XB, tB, same=False):
        K = np.zeros((len(XA), len(XB)))
        for B, w in zip(self.Bs_, self.ws_):
            K += B[np.ix_(tA, tB)] * _kx(XA, XB, w, same=same)
        return K

    def _noise(self, tA, rA, tB, rB):
        return self.S_[np.ix_(tA, tB)] * (rA[:, None] == rB[None, :])

    def _condition(self, Xo, to, yo, ro):
        K = self._cov(Xo, to, Xo, to, same=True) + self._noise(to, ro, to, ro)
        self.C_ = cho_factor(K + 1e-8 * np.eye(len(K)), lower=True)
        self.alpha_ = cho_solve(self.C_, yo)
        self.cond_ = (Xo, to, ro)

    # ------------------------------------------------------------------ predict
    def _predict_z(self, Xz, return_std, include_noise, row_ids=None):
        """row_ids: ids of the rows being predicted; where they match conditioning observations of the
        same row, the (correlated) noise covariance links them (only when predicting noisy outputs)."""
        Xo, to, ro = self.cond_
        M = self.n_tasks_
        mean = np.empty((len(Xz), M))
        sd = np.empty_like(mean)
        prior_var = sum(np.diag(B) for B in self.Bs_)
        rid = np.full(len(Xz), -1) if row_ids is None else np.asarray(row_ids)
        for t in range(M):
            tt = np.full(len(Xz), t)
            Ks = self._cov(Xz, tt, Xo, to)
            if include_noise:
                Ks = Ks + self._noise(tt, rid, to, ro)
            mean[:, t] = Ks @ self.alpha_
            if return_std:
                v = cho_solve(self.C_, Ks.T)
                var = prior_var[t] - (Ks * v.T).sum(1) + (self.n2_[t] if include_noise else 0.0)
                sd[:, t] = np.sqrt(np.maximum(var, 0))
        return mean, sd

    def predict(self, X, return_std: bool = False, include_noise: bool = True):
        """Predictions for every output from the inputs alone: arrays of shape (rows, outputs)."""
        mean, sd = self._predict_z(self.xs_.transform(np.asarray(X, dtype=np.float64)), return_std, include_noise)
        mean = self.ys_.inverse_transform(mean)
        return (mean, sd * self.ys_.scale_) if return_std else mean

    def predict_given(self, X, Y_partial, return_std: bool = False, include_noise: bool = True):
        """Predictions for every output of new rows, conditioning on the outputs they already have
        (non-NaN entries of Y_partial). Known entries are returned as predicted by the model, not copied."""
        X = np.asarray(X, dtype=np.float64)
        Yp = np.asarray(Y_partial, dtype=np.float64)
        Yp = Yp[:, None] if Yp.ndim == 1 else Yp
        Xz = self.xs_.transform(X)
        rows, tasks = np.nonzero(~np.isnan(Yp))
        new_ids = self.n_train_ + np.arange(len(Xz))           # each new row is its own "patient"
        saved = (self.C_, self.alpha_, self.cond_)
        try:
            if len(rows):
                Xo = np.vstack([self.Xo_, Xz[rows]])
                to = np.concatenate([self.to_, tasks])
                yo = np.concatenate([self.yo_, self.ys_.transform(Yp)[rows, tasks]])
                ro = np.concatenate([self.ro_, new_ids[rows]])
                self._condition(Xo, to, yo, ro)
            mean, sd = self._predict_z(Xz, return_std, include_noise, row_ids=new_ids)
        finally:
            self.C_, self.alpha_, self.cond_ = saved
        mean = self.ys_.inverse_transform(mean)
        return (mean, sd * self.ys_.scale_) if return_std else mean

    # ------------------------------------------------------------------ inspection
    def task_covariance(self) -> np.ndarray:
        """Total between-output covariance of the latent functions (sum of B_q), in z-units."""
        return sum(self.Bs_)

    def task_correlation(self) -> np.ndarray:
        B = self.task_covariance()
        d = np.sqrt(np.diag(B))
        return B / np.outer(d, d)

    def relevance(self) -> np.ndarray:
        """(features, outputs): ARD weights of each latent function, weighted by how much of each
        output's variance it carries. Larger = more relevant for that output."""
        share = np.array([np.diag(B) for B in self.Bs_])            # (Q, outputs)
        share = share / share.sum(0, keepdims=True)
        return np.array(self.ws_).T @ share


def predict_trajectory(model, x_row, times, time_column: int = 0, y_known=None):
    """Predicted outputs for one patient over time (MGP_Bonilla_Prog.m): the patient's inputs are
    repeated with the time input set to each value in `times`.

    y_known: the patient's observed outputs (NaN where unknown) to condition on, if the model can.
    Returns (mean, sd) arrays of shape (len(times), outputs); sd is None if the model gives none.

    MGP_Bonilla_Prog.m stored each patient's prediction in one column (Ypred(:, k)), which fails with
    more than one output, returned variances rather than SDs, and scaled inputs by their maximum."""
    x = np.tile(np.asarray(x_row, dtype=np.float64).ravel(), (len(times), 1))
    x[:, time_column] = np.asarray(times, dtype=np.float64)
    if y_known is not None:
        yk = np.tile(np.asarray(y_known, dtype=np.float64).ravel(), (len(times), 1))
        out = model.predict_given(x, yk, return_std=True)
    else:
        out = model.predict(x, return_std=True)
    return out if isinstance(out, tuple) else (out, None)

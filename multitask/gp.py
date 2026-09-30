"""Single-output Gaussian-process regression with automatic relevance determination (ARD),
and SingleOutputGPs: one independent GP per output, the "multiple single-output models" that the
multi-task GP is compared against.

(The GP is the same one as in the ploras package, the OptimisePredictions rewrite.)

Kernel (the original used Netlab's 'ratquad'):

    k(x, x') = s2 * (1 + d2 / (2 a)) ** (-a)          'ratquad'   (rational quadratic)
             = s2 * exp(-d2 / 2)                       'sqexp'     (squared exponential)
             = s2 * exp(-sqrt(d2))                     'exponential' (fitrgp's 'exponential')
    d2 = sum_i w_i (x_i - x'_i)^2,   plus a constant (bias) b2 and noise n2 on the diagonal.

w_i are the ARD weights (inverse squared length scales): a large w_i means the
outcome changes quickly with feature i, i.e. the feature is relevant. RunARD ranked
features by exp(gpnet.inweights), which are the same thing.

Hyperparameters are fitted by maximising the log marginal likelihood (L-BFGS with
analytic gradients) plus, by default, a N(0, 1) prior on every log-hyperparameter,
as Netlab's gpinit(..., prior) with pr_mean 0 and pr_var 1 did. Instead of the
original's "add jitter to the inputs and retry" loop, a failed Cholesky gets a
growing diagonal jitter and the optimiser is restarted from several points.

Inputs and outputs are z-scored internally with the training data's statistics, so
predictions come back in the outcome's own units.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

from .stats import Standardiser

KERNELS = ("ratquad", "sqexp", "exponential")


class GaussianProcess:
    def __init__(self, kernel: str = "ratquad", ard: bool = True, prior_var: float | None = 1.0,
                 n_restarts: int = 2, max_iter: int = 200, standardise: bool = True, seed: int | None = 0):
        if kernel not in KERNELS:
            raise ValueError(f"kernel must be one of {KERNELS}")
        self.kernel, self.ard, self.prior_var = kernel, ard, prior_var
        self.n_restarts, self.max_iter, self.standardise, self.seed = n_restarts, max_iter, standardise, seed

    # ------------------------------------------------------------------ parameters
    # theta = [log s2, (log a if ratquad), log w (D or 1), log b2, log n2]
    def _unpack(self, theta):
        i = 0
        out = {"s2": np.exp(theta[i])}
        i += 1
        if self.kernel == "ratquad":
            out["a"] = np.exp(theta[i])
            i += 1
        nw = self.n_features_ if self.ard else 1
        out["w"] = np.exp(theta[i:i + nw]) * np.ones(self.n_features_)
        i += nw
        out["b2"], out["n2"] = np.exp(theta[i]), np.exp(theta[i + 1])
        return out

    def _initial(self, rng):
        nw = self.n_features_ if self.ard else 1
        parts = [[0.0]] + ([[0.0]] if self.kernel == "ratquad" else []) + \
                [np.full(nw, -np.log(self.n_features_))] + [[-2.0], [-1.0]]
        theta = np.concatenate(parts)
        if rng is not None:
            theta = theta + rng.normal(scale=0.5, size=theta.shape)
        return theta

    # ------------------------------------------------------------------ kernel
    def _d2(self, A, B, w):
        Aw = A * np.sqrt(w)
        Bw = B * np.sqrt(w)
        d2 = np.maximum((Aw ** 2).sum(1)[:, None] + (Bw ** 2).sum(1)[None, :] - 2 * Aw @ Bw.T, 0.0)
        if A is B:
            np.fill_diagonal(d2, 0.0)       # exact zeros on the diagonal (matters for 'exponential')
        return d2

    def _k(self, d2, p):
        if self.kernel == "ratquad":
            return p["s2"] * (1 + d2 / (2 * p["a"])) ** (-p["a"])
        if self.kernel == "sqexp":
            return p["s2"] * np.exp(-d2 / 2)
        return p["s2"] * np.exp(-np.sqrt(d2))

    def _nlml(self, theta, X, y):
        p = self._unpack(theta)
        n = len(y)
        d2 = self._d2(X, X, p["w"])
        Kf = self._k(d2, p)
        K = Kf + p["b2"] + p["n2"] * np.eye(n)
        jitter = 0.0
        if not np.isfinite(K).all():
            return 1e25, np.zeros_like(theta)
        for _ in range(6):
            try:
                L = cho_factor(K + jitter * np.eye(n), lower=True)
                break
            except (np.linalg.LinAlgError, ValueError):
                jitter = 1e-8 if jitter == 0 else jitter * 100
        else:
            return 1e25, np.zeros_like(theta)
        alpha = cho_solve(L, y)
        logdet = 2 * np.log(np.diag(L[0])).sum()
        nll = 0.5 * y @ alpha + 0.5 * logdet + 0.5 * n * np.log(2 * np.pi)
        Kinv = cho_solve(L, np.eye(n))
        W = np.outer(alpha, alpha) - Kinv          # d(-nll)/dK = W/2

        grads = []
        grads.append(Kf)                                                 # d/dlog s2
        if self.kernel == "ratquad":
            a = p["a"]
            u = 1 + d2 / (2 * a)
            grads.append(Kf * (-a * np.log(u) + d2 / (2 * u)))           # d/dlog a
            dk_dd2 = -Kf / (2 * u)
        elif self.kernel == "sqexp":
            dk_dd2 = -Kf / 2
        else:
            r = np.sqrt(d2)
            with np.errstate(divide="ignore", invalid="ignore"):
                dk_dd2 = np.where(r > 0, -Kf / (2 * r), 0.0)
        g = np.empty(len(theta))
        g[0] = -0.5 * (W * grads[0]).sum()
        i = 1
        if self.kernel == "ratquad":
            g[1] = -0.5 * (W * grads[1]).sum()
            i = 2
        WK = W * dk_dd2
        if self.ard:
            for d in range(self.n_features_):
                diff2 = (X[:, d][:, None] - X[:, d][None, :]) ** 2
                g[i + d] = -0.5 * (WK * diff2 * p["w"][d]).sum()
            i += self.n_features_
        else:
            g[i] = -0.5 * (WK * d2).sum()
            i += 1
        g[i] = -0.5 * W.sum() * p["b2"]
        g[i + 1] = -0.5 * np.trace(W) * p["n2"]
        if self.prior_var is not None:
            nll += 0.5 * (theta ** 2).sum() / self.prior_var
            g += theta / self.prior_var
        return nll, g

    # ------------------------------------------------------------------ fit / predict
    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        if X.ndim == 1:
            X = X[:, None]
        if np.isnan(X).any() or np.isnan(y).any():
            raise ValueError("remove rows with missing values before fitting the GP")
        self.n_features_ = X.shape[1]
        self.xs_ = Standardiser().fit(X) if self.standardise else None
        self.ys_ = Standardiser().fit(y[:, None]) if self.standardise else None
        Xz = self.xs_.transform(X) if self.standardise else X
        yz = self.ys_.transform(y[:, None]).ravel() if self.standardise else y
        rng = np.random.default_rng(self.seed)
        best = None
        for r in range(max(1, self.n_restarts)):
            theta0 = self._initial(None if r == 0 else rng)
            res = minimize(self._nlml, theta0, args=(Xz, yz), jac=True, method="L-BFGS-B",
                           bounds=[(-15.0, 15.0)] * len(theta0), options={"maxiter": self.max_iter})
            if best is None or res.fun < best.fun:
                best = res
        self.theta_ = best.x
        self.nlml_ = float(best.fun)
        self.params_ = self._unpack(best.x)
        self.X_, self.y_ = Xz, yz
        n = len(yz)
        K = self._k(self._d2(Xz, Xz, self.params_["w"]), self.params_) + self.params_["b2"] + \
            self.params_["n2"] * np.eye(n)
        jitter = 0.0
        while True:
            try:
                self.L_ = cho_factor(K + jitter * np.eye(n), lower=True)
                break
            except np.linalg.LinAlgError:
                jitter = 1e-8 if jitter == 0 else jitter * 100
        self.alpha_ = cho_solve(self.L_, yz)
        return self

    def predict(self, X, return_std: bool = False, include_noise: bool = True):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X[:, None]
        Xz = self.xs_.transform(X) if self.standardise else X
        p = self.params_
        Ks = self._k(self._d2(Xz, self.X_, p["w"]), p) + p["b2"]
        mean = Ks @ self.alpha_
        if not return_std:
            return self._unscale_mean(mean)
        v = cho_solve(self.L_, Ks.T)
        var = p["s2"] + p["b2"] - (Ks * v.T).sum(1)
        if include_noise:
            var = var + p["n2"]
        sd = np.sqrt(np.maximum(var, 0.0))
        return self._unscale_mean(mean), sd * (self.ys_.scale_[0] if self.standardise else 1.0)

    def _unscale_mean(self, m):
        return self.ys_.inverse_transform(m[:, None]).ravel() if self.standardise else m

    @property
    def relevance_(self) -> np.ndarray:
        """ARD weights (on z-scored features): larger = more relevant."""
        return self.params_["w"].copy()

    def feature_ranking(self) -> np.ndarray:
        """Feature indices, most relevant first (RunARD's D column)."""
        return np.argsort(-self.relevance_, kind="stable")

    # sklearn-style plumbing, so it can be used wherever the other models are
    def get_params(self, deep=True):
        return dict(kernel=self.kernel, ard=self.ard, prior_var=self.prior_var, n_restarts=self.n_restarts,
                    max_iter=self.max_iter, standardise=self.standardise, seed=self.seed)

    def set_params(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)
        return self


class SingleOutputGPs:
    """One independent GP per output column; each is fitted on the rows where its output is present.

    Missing outputs (NaN in Y) are fine: each GP simply has fewer rows. Nothing is shared between
    outputs, which is the point of the comparison."""

    def __init__(self, kernel: str = "sqexp", ard: bool = True, prior_var: float | None = None, **gp_args):
        """Squared-exponential ARD kernel (the multi-task GP's kernel) and no hyperparameter prior
        by default, so that the two are compared like for like."""
        self.kernel, self.ard, self.gp_args = kernel, ard, dict(gp_args, prior_var=prior_var)

    def fit(self, X, Y):
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        Y = Y[:, None] if Y.ndim == 1 else Y
        self.models_ = []
        for j in range(Y.shape[1]):
            ok = ~np.isnan(Y[:, j])
            self.models_.append(GaussianProcess(kernel=self.kernel, ard=self.ard, **self.gp_args).fit(X[ok], Y[ok, j]))
        return self

    def predict(self, X, return_std: bool = False):
        out = [m.predict(X, return_std=True) for m in self.models_]
        mean = np.column_stack([o[0] for o in out])
        if not return_std:
            return mean
        return mean, np.column_stack([o[1] for o in out])

    def predict_given(self, X, Y_partial, return_std: bool = False):
        """Independent models cannot use a patient's other known outputs, so this is predict()."""
        return self.predict(X, return_std)

    def relevance(self) -> np.ndarray:
        """(features, outputs) ARD weights; larger = more relevant for that output."""
        return np.column_stack([m.relevance_ for m in self.models_])

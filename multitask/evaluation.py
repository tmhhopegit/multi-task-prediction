"""Paired comparison of multi-output and single-output models by cross-validation.

Replaces CVGP_Bonilla.m, CV_TenXTen_MultiGP / CV_MultiGP (MultiGP*.m), TestMultiTaskLearning.m,
BonPred.m and the Legion_* job scripts (which split the same cross-validation into cluster jobs;
here folds run in parallel with n_jobs).

Two prediction modes:

    "from_inputs"  predict every output of a new patient from their inputs alone
                   (what the original code did)
    "fill_in"      predict each output of a held-out patient from their inputs AND their other
                   outputs, e.g. a test score they did not take from the ones they did. This is
                   where a multi-output model can use what independent models cannot.

Every model sees exactly the same folds, so differences are paired. Inputs and outputs are scaled
inside each training fold (the original z-scored all patients before cross-validating).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold

from .models import KIND, make_model
from .stats import gaussian_nll, regression_scores

MODES = ("from_inputs", "fill_in")


def make_folds(n: int, k: int = 10, repeats: int = 1, groups=None, seed: int | None = 0):
    """[(repeat, train, test)]; k >= n without groups is leave-one-out (once)."""
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    if groups is None and k >= n:
        return [(0, np.delete(idx, i), np.array([i])) for i in range(n)]
    out = []
    for r in range(repeats):
        if groups is None:
            split = KFold(k, shuffle=True, random_state=int(rng.integers(2**31))).split(idx)
        else:
            g = np.asarray(groups)
            uniq = np.unique(g)
            relabel = dict(zip(uniq, rng.permutation(len(uniq))))
            split = GroupKFold(min(k, len(uniq))).split(idx, groups=np.array([relabel[v] for v in g]))
        out += [(r, tr, te) for tr, te in split]
    return out


def _run_fold(name, model_args, X, Y, tr, te, mode, seed):
    model = make_model(name, seed=seed, **model_args.get(name, {}))
    model.fit(X[tr], Y[tr])
    M = Y.shape[1]
    std = None
    if mode == "from_inputs":
        out = model.predict(X[te], return_std=True)
        pred, std = out if isinstance(out, tuple) else (out, None)
    else:
        pred = np.full((len(te), M), np.nan)
        std = np.full((len(te), M), np.nan)
        for j in range(M):
            Yp = Y[te].copy()
            Yp[:, j] = np.nan
            out = model.predict_given(X[te], Yp, return_std=True)
            p, s = out if isinstance(out, tuple) else (out, None)
            if s is not None:
                std[:, j] = np.asarray(s)[:, j]
            pred[:, j] = np.asarray(p)[:, j]
        if np.isnan(std).all():
            std = None
    return np.asarray(pred, dtype=np.float64), None if std is None else np.asarray(std, dtype=np.float64)


@dataclass
class CVResults:
    Y: np.ndarray
    models: list
    mode: str
    predictions: dict = field(default_factory=dict)   # name -> (rows, outputs, repeats)
    stds: dict = field(default_factory=dict)
    output_names: list | None = None

    def mean_prediction(self, name):
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                return np.nanmean(self.predictions[name], axis=2)

    def scores(self) -> pd.DataFrame:
        """One row per model x output: r, R2, MAE, RMSE (and NLL for GPs)."""
        rows = []
        names = self.output_names or [f"output_{j}" for j in range(self.Y.shape[1])]
        for m in self.models:
            P = self.mean_prediction(m)
            for j in range(self.Y.shape[1]):
                s = regression_scores(self.Y[:, j], P[:, j])
                if m in self.stds:
                    with np.errstate(all="ignore"):
                        import warnings
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", RuntimeWarning)
                            sd = np.sqrt(np.nanmean(self.stds[m][:, j, :] ** 2, axis=1))
                    ok = np.isfinite(sd) & np.isfinite(P[:, j]) & np.isfinite(self.Y[:, j])
                    s["nll"] = float(gaussian_nll(self.Y[ok, j], P[ok, j], sd[ok] ** 2).mean()) if ok.any() else np.nan
                rows.append({"model": m, "kind": KIND.get(m, "?"), "output": names[j], **s})
        return pd.DataFrame(rows)

    def compare(self, single: str, multi: str, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
        """Per output: mean squared error of `single` minus that of `multi` (positive = multi better),
        with a 95% bootstrap interval over patients; plus the same pooled over outputs."""
        rng = np.random.default_rng(seed)
        A, B = self.mean_prediction(single), self.mean_prediction(multi)
        names = self.output_names or [f"output_{j}" for j in range(self.Y.shape[1])]
        rows = []
        diffs_all = []
        for j in range(self.Y.shape[1]):
            ok = np.isfinite(A[:, j]) & np.isfinite(B[:, j]) & np.isfinite(self.Y[:, j])
            d = (A[ok, j] - self.Y[ok, j]) ** 2 - (B[ok, j] - self.Y[ok, j]) ** 2
            diffs_all.append((np.flatnonzero(ok), d / np.var(self.Y[ok, j])))
            boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)]
            lo, hi = np.percentile(boots, [2.5, 97.5])
            rows.append({"output": names[j], "single": single, "multi": multi, "mse_single": float(((A[ok, j] - self.Y[ok, j]) ** 2).mean()),
                         "mse_multi": float(((B[ok, j] - self.Y[ok, j]) ** 2).mean()), "difference": float(d.mean()),
                         "ci_low": float(lo), "ci_high": float(hi),
                         "verdict": "multi better" if lo > 0 else ("single better" if hi < 0 else "no clear difference")})
        # pooled (each output's squared errors scaled by its variance), bootstrapping patients
        n = len(self.Y)
        per_row = np.full((n, self.Y.shape[1]), np.nan)
        for j, (idx, d) in enumerate(diffs_all):
            per_row[idx, j] = d
        row_mean = np.nanmean(per_row, axis=1)
        row_mean = row_mean[np.isfinite(row_mean)]
        boots = [row_mean[rng.integers(0, len(row_mean), len(row_mean))].mean() for _ in range(n_boot)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        rows.append({"output": "all (variance-scaled)", "single": single, "multi": multi, "mse_single": np.nan,
                     "mse_multi": np.nan, "difference": float(row_mean.mean()), "ci_low": float(lo), "ci_high": float(hi),
                     "verdict": "multi better" if lo > 0 else ("single better" if hi < 0 else "no clear difference")})
        return pd.DataFrame(rows)


def cross_validate(models, X, Y, k: int = 10, repeats: int = 1, groups=None, mode: str = "from_inputs",
                   seed: int = 0, n_jobs: int = 1, model_args: dict | None = None, output_names=None,
                   verbose: bool = False) -> CVResults:
    """Cross-validated predictions for every model on the same folds.

    Rows with missing inputs are dropped. Missing outputs are allowed: models fit on what is
    observed, and only observed values are scored."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    Y = Y[:, None] if Y.ndim == 1 else Y
    ok = ~np.isnan(X).any(1) & ~np.isnan(Y).all(1)
    X, Y = X[ok], Y[ok]
    groups = None if groups is None else np.asarray(groups)[ok]
    models = list(models)
    model_args = model_args or {}
    folds = make_folds(len(X), k, repeats, groups, seed)
    n_rep = max(r for r, _, _ in folds) + 1
    jobs = [(name, r, tr, te) for name in models for (r, tr, te) in folds]

    def run(job):
        name, r, tr, te = job
        return job, _run_fold(name, model_args, X, Y, tr, te, mode, seed + r)

    if n_jobs == 1:
        outs = []
        for i, j in enumerate(jobs):
            outs.append(run(j))
            if verbose:
                print(f"{i + 1}/{len(jobs)} {j[0]}", flush=True)
    else:
        from joblib import Parallel, delayed
        outs = Parallel(n_jobs=n_jobs)(delayed(run)(j) for j in jobs)
    res = CVResults(Y, models, mode, output_names=output_names)
    for name in models:
        res.predictions[name] = np.full(Y.shape + (n_rep,), np.nan)
    for (name, r, tr, te), (pred, std) in outs:
        res.predictions[name][te, :, r] = pred
        if std is not None:
            res.stds.setdefault(name, np.full(Y.shape + (n_rep,), np.nan))[te, :, r] = std
    return res

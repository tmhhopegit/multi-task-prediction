"""Choosing input features for multi-output models.

    ard_rankings        rank features separately for each output by GP relevance (RunARD in the
                        original; each column of D)
    forward_selection   the scheme of MultiGP.m / MultiGP_MassiveParallel.m / Legion_*: start from a
                        few features; at each step the candidates are the next-ranked unused feature
                        of each output's ranking; add the one that most improves the cross-validated
                        criterion
    size_scan           cross-validated performance of the top-d features of one ranking, for each d
                        (PLORAS_MultiGP_V2.m)

Fixes relative to the original forward selection:
- It stops when no candidate improves the criterion (stop_when_worse=True). The original's
  `Improving` flag was never set by performance: it ran until the model exceeded MaxLim.
- A candidate already in the model is skipped. The original could offer a feature that another
  output's ranking had already added, testing a model with a duplicated column.
- The criterion is fixed. The original alternated between the mean and the maximum correlation
  over outputs on odd and even steps; that is available as criterion="alternate".
- The history of every step is returned. MultiGP.m returned R and E exactly as initialised
  (-100 and Inf).
- Scaling happens inside the cross-validation folds (the original z-scored all patients first).

A selection's own cross-validated score is optimistic (it is the best of many tries); nested()
gives an honest estimate by repeating the whole selection inside an outer cross-validation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .evaluation import cross_validate, make_folds
from .gp import SingleOutputGPs
from .models import make_model
from .stats import regression_scores


def ard_rankings(X, Y, first=None) -> np.ndarray:
    """(features, outputs): feature indices for each output, most relevant first. `first`: features
    forced to the top for every output (the original always put column 1, time post-stroke, first)."""
    rel = SingleOutputGPs().fit(X, Y).relevance()
    D = np.argsort(-rel, axis=0, kind="stable")
    if first is not None:
        first = list(np.atleast_1d(first))
        D = np.column_stack([first + [f for f in D[:, j] if f not in first] for j in range(D.shape[1])])
    return D


def _criterion(scores_by_output, criterion, step):
    r = np.array([s["r"] for s in scores_by_output])
    if criterion == "mean_r":
        return float(np.nanmean(r))
    if criterion == "max_r":
        return float(np.nanmax(r))
    if criterion == "alternate":                       # the original: max on even model sizes, mean on odd
        return float(np.nanmax(r)) if step % 2 == 0 else float(np.nanmean(r))
    if criterion == "mean_R2":
        return float(np.nanmean([s["R2"] for s in scores_by_output]))
    raise ValueError("criterion must be mean_r, max_r, mean_R2 or alternate")


@dataclass
class SelectionPath:
    features: list = field(default_factory=list)     # the model after each step
    criterion: list = field(default_factory=list)
    scores: list = field(default_factory=list)       # per-output scores after each step

    def best(self):
        i = int(np.argmax(self.criterion))
        return self.features[i], self.criterion[i]


def forward_selection(model: str, X, Y, rankings, start=None, criterion: str = "mean_r", max_features: int = 20,
                      stop_when_worse: bool = True, min_gain: float = 0.005, k: int = 10, repeats: int = 1,
                      seed: int = 0, model_args: dict | None = None, verbose: bool = False) -> SelectionPath:
    """Forward selection over the outputs' rankings (see the module docstring).

    start: initial features (default: the top-ranked feature of every output; MultiGP.m started from
    feature 1, MultiGP_MassiveParallel.m from the top 5 of every output).
    min_gain: with stop_when_worse, stop unless the best candidate improves the criterion by at least
    this much (cross-validated scores fluctuate, so tiny gains usually mean an irrelevant feature)."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    Y = Y[:, None] if Y.ndim == 1 else Y
    rankings = np.asarray(rankings)
    selected = list(dict.fromkeys(start if start is not None else rankings[0].tolist()))
    path = SelectionPath()

    def evaluate(feats):
        res = cross_validate([model], X[:, feats], Y, k=k, repeats=repeats, seed=seed, model_args=model_args)
        P = res.mean_prediction(model)
        return [regression_scores(Y[:, j], P[:, j]) for j in range(Y.shape[1])]

    sc = evaluate(selected)
    current = _criterion(sc, criterion, len(selected))
    path.features.append(list(selected)); path.criterion.append(current); path.scores.append(sc)
    while len(selected) < max_features:
        candidates = []
        for j in range(rankings.shape[1]):
            nxt = next((f for f in rankings[:, j] if f not in selected), None)
            if nxt is not None and nxt not in candidates:
                candidates.append(int(nxt))
        if not candidates:
            break
        results = []
        for c in candidates:
            s = evaluate(selected + [c])
            results.append((_criterion(s, criterion, len(selected) + 1), c, s))
        best_val, best_c, best_s = max(results, key=lambda t: t[0])
        if stop_when_worse and best_val < current + min_gain:
            break
        selected.append(best_c)
        current = best_val
        path.features.append(list(selected)); path.criterion.append(best_val); path.scores.append(best_s)
        if verbose:
            print(f"{len(selected):3d} features (+{best_c}): {criterion} = {best_val:.4f}", flush=True)
    return path


def size_scan(model: str, X, Y, ranking, sizes, k: int = 10, repeats: int = 1, seed: int = 0,
              model_args: dict | None = None) -> list:
    """[(d, per-output scores)] for the top-d features of `ranking` (PLORAS_MultiGP_V2.m)."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    Y = Y[:, None] if Y.ndim == 1 else Y
    out = []
    for d in sizes:
        res = cross_validate([model], X[:, list(ranking[:d])], Y, k=k, repeats=repeats, seed=seed,
                             model_args=model_args)
        P = res.mean_prediction(model)
        out.append((int(d), [regression_scores(Y[:, j], P[:, j]) for j in range(Y.shape[1])]))
    return out


def nested(model: str, X, Y, max_features: int = 10, criterion: str = "mean_r", k_outer: int = 5, k_inner: int = 5,
           seed: int = 0, model_args: dict | None = None) -> dict:
    """Out-of-sample performance of "rank features, forward-select, then fit": rankings and selection
    are redone on each outer training set, and the held-out rows are predicted with the chosen features."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    Y = Y[:, None] if Y.ndim == 1 else Y
    P = np.full(Y.shape, np.nan)
    chosen = []
    for _, tr, te in make_folds(len(X), k_outer, 1, None, seed):
        D = ard_rankings(X[tr], Y[tr])
        path = forward_selection(model, X[tr], Y[tr], D, criterion=criterion, max_features=max_features, k=k_inner,
                                 seed=seed, model_args=model_args)
        feats = path.best()[0]
        chosen.append(feats)
        m = make_model(model, seed=seed, **(model_args or {}).get(model, {})).fit(X[tr][:, feats], Y[tr])
        P[te] = m.predict(X[te][:, feats])
    return {"predictions": P, "scores": [regression_scores(Y[:, j], P[:, j]) for j in range(Y.shape[1])],
            "features_per_fold": chosen}

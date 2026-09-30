"""A benchmark on synthetic data: when do multi-output models beat separate single-output ones?

Scenarios (each run on several independently generated datasets):

    toy_1d            the original TestMultiTaskLearning.m problem (1 input, 3 outputs), all observed
    related           4 outputs sharing 80-90% of their signal, all observed
    related_small     the same with only 60 patients
    related_missing   the same with 40% of outputs missing at random, predicted from inputs only
    related_fill_in   40% missing; each held-out output predicted from the inputs AND the patient's
                      other outputs
    unrelated_fill_in outputs with nothing in common, 40% missing, fill-in mode (multi-output models
                      should gain nothing here, and ideally lose nothing)
    patient_fill_in   related outputs plus a patient-level factor (e.g. overall severity) that the
                      inputs do not explain; 40% missing, fill-in mode. Only a patient's own other
                      outputs carry that information.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .evaluation import cross_validate
from .synthetic import mask_missing, shared_factors, toy_1d

SCENARIOS = {
    "toy_1d": dict(kind="toy", mode="from_inputs"),
    "related": dict(n=150, relatedness=0.9, missing=0.0, mode="from_inputs"),
    "related_small": dict(n=60, relatedness=0.9, missing=0.0, mode="from_inputs"),
    "related_missing": dict(n=150, relatedness=0.9, missing=0.4, mode="from_inputs"),
    "related_fill_in": dict(n=150, relatedness=0.9, missing=0.4, mode="fill_in"),
    "unrelated_fill_in": dict(n=150, relatedness=0.0, missing=0.4, mode="fill_in"),
    "patient_fill_in": dict(n=150, relatedness=0.9, missing=0.4, mode="fill_in", patient_factor=1.0),
}
DEFAULT_MODELS = ["gp", "mtgp", "mtgp_corr", "gp_augmented", "ridge", "rrr", "lasso", "mtlasso", "pls1", "pls2"]
PAIRS = [("gp", "mtgp"), ("gp", "mtgp_corr"), ("gp_augmented", "mtgp"), ("gp_augmented", "mtgp_corr"), ("ridge", "rrr"), ("lasso", "mtlasso"), ("pls1", "pls2")]


def make_data(spec, seed):
    if spec.get("kind") == "toy":
        X, Y = toy_1d(100, seed=seed)
        return X, Y
    X, Y, _ = shared_factors(n=spec["n"], relatedness=spec["relatedness"],
                             patient_factor=spec.get("patient_factor", 0.0), seed=seed)
    return X, (mask_missing(Y, spec["missing"], seed=seed + 1000) if spec["missing"] else Y)


def run(scenarios=None, models=None, datasets: int = 3, k: int = 5, n_jobs: int = 1, verbose: bool = True):
    """Returns (scores: one row per scenario x dataset x model x output, comparisons: one row per
    scenario x dataset x pair, pooled over outputs)."""
    scenarios = scenarios or list(SCENARIOS)
    models = models or DEFAULT_MODELS
    score_rows, comp_rows = [], []
    for name in scenarios:
        spec = SCENARIOS[name]
        use = [m for m in models if not (m == "gp_augmented" and spec["mode"] != "fill_in")]
        for d in range(datasets):
            t0 = time.time()
            X, Y = make_data(spec, seed=d)
            res = cross_validate(use, X, Y, k=k, mode=spec["mode"], seed=d, n_jobs=n_jobs)
            sc = res.scores()
            sc.insert(0, "dataset", d)
            sc.insert(0, "scenario", name)
            score_rows.append(sc)
            for single, multi in PAIRS:
                if single in use and multi in use:
                    c = res.compare(single, multi, n_boot=1000, seed=d).iloc[-1]
                    comp_rows.append({"scenario": name, "dataset": d, "single": single, "multi": multi,
                                      "difference": c.difference, "ci_low": c.ci_low, "ci_high": c.ci_high,
                                      "verdict": c.verdict})
            if verbose:
                print(f"{name} dataset {d}: {time.time() - t0:.0f}s", flush=True)
    return pd.concat(score_rows, ignore_index=True), pd.DataFrame(comp_rows)


def summarise(scores: pd.DataFrame) -> pd.DataFrame:
    """Mean R2 over datasets and outputs, scenario x model."""
    return scores.groupby(["scenario", "model"]).R2.mean().unstack("model")


def figure(scores: pd.DataFrame, path=None):
    """Small multiples, one per model family: out-of-sample R2 of the single-output (blue) and
    multi-output (orange) version in each scenario, one dot per dataset."""
    import matplotlib.pyplot as plt
    ink, ink2, grid, surface = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    col = {"single": "#2a78d6", "multi": "#eb6834"}
    families = [("GP vs MTGP", "gp", "mtgp"), ("GP vs MTGP, correlated noise", "gp", "mtgp_corr"),
                ("GP, other outputs as inputs, vs MTGP corr.", "gp_augmented", "mtgp_corr"), ("Ridge", "ridge", "rrr"),
                ("Lasso", "lasso", "mtlasso"), ("PLS", "pls1", "pls2")]
    order = [s for s in SCENARIOS if s in set(scores.scenario)]
    per = scores.groupby(["scenario", "dataset", "model"]).R2.mean().reset_index()
    xmin = -0.5
    ncol = 3
    nrow = int(np.ceil(len(families) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, nrow * (0.36 * len(order) + 1.9)),
                             sharey=True, sharex=True, facecolor=surface, squeeze=False)
    for ax in axes.ravel()[len(families):]:
        ax.set_visible(False)
    for ax, (title, s, m) in zip(axes.ravel(), families):
        ax.set_facecolor(surface)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(grid)
        ax.grid(True, axis="x", color=grid, lw=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(colors=ink2, labelsize=8)
        for kind, model, off in (("single", s, -0.15), ("multi", m, 0.15)):
            labelled = False
            for i, sc in enumerate(order):
                v = per[(per.scenario == sc) & (per.model == model)].R2
                if len(v):
                    v = v.to_numpy()
                    low = v < xmin
                    ax.scatter(v[~low], np.full((~low).sum(), i + off), s=22, color=col[kind], edgecolor=surface,
                               lw=1.2, zorder=3, label=None if labelled else model)
                    if low.any():                     # below the axis: drawn at the edge as an arrow head
                        ax.scatter(np.full(low.sum(), xmin), np.full(low.sum(), i + off), marker="<", s=26,
                                   color=col[kind], zorder=3)
                    labelled = True
        ax.set_xlim(xmin - 0.04, 1.0)
        ax.tick_params(labelbottom=True)
        ax.set_title(title, fontsize=9, color=ink)
        ax.set_xlabel("out-of-sample R²", fontsize=8, color=ink2)
        ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    axes[0, 0].set_yticks(range(len(order)))
    axes[0, 0].set_yticklabels(order, fontsize=8)
    axes[0, 0].invert_yaxis()
    for ax in axes[:, 0]:
        ax.tick_params(labelleft=True)
    fig.suptitle("Single-output (blue) vs multi-output (orange) models on synthetic data; one dot per dataset"
                 f" (arrow heads: R² below {xmin})",
                 fontsize=10, color=ink, x=0.01, ha="left")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, facecolor=surface, bbox_inches="tight")
    return fig

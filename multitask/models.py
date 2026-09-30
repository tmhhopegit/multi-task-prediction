"""Every model by name, and the single-output / multi-output pairs being compared.

    name            model                                              kind
    gp              one ARD GP per output                              single
    mtgp            multi-task GP, 1 latent function (Bonilla / ICM)   multi
    mtgp_corr       mtgp with correlated noise between one row's outputs  multi
    lmc             multi-task GP, 2 latent functions (LMC)            multi
    gp_augmented    one GP per output, other outputs as extra inputs   single (uses other outputs)
    ridge           ridge per output                                   single
    rrr             reduced-rank ridge                                 multi
    lasso           lasso per output                                   single
    mtlasso         multi-task lasso (L2,1)                            multi
    pls1            PLS per output                                     single
    pls2            one PLS for all outputs                            multi
    pca_mtgp        PCA (in-fold), then mtgp on the components         multi  (RunPredictivePLS.m)
    pca_gp          PCA (in-fold), then one GP per output              single
    peak_voxel      regression on the most correlated feature          single (RunPredictivePLS.m's comment)

All have fit(X, Y) with NaN allowed in Y, predict(X), and predict_given(X, Y_partial), which uses
a new row's known outputs where the model can (mtgp, lmc, gp_augmented) and ignores them otherwise.
GP models also give predict(X, return_std=True).
"""
from __future__ import annotations

from .gp import SingleOutputGPs
from .linear import (AugmentedSingle, IndependentLasso, IndependentRidge, MultiTaskLasso, PLS1, PLS2,
                     ReducedRankRidge)
from .mtgp import MultiTaskGP

KIND = {"gp": "single", "mtgp": "multi", "mtgp_corr": "multi", "lmc": "multi", "gp_augmented": "single", "ridge": "single",
        "rrr": "multi", "lasso": "single", "mtlasso": "multi", "pls1": "single", "pls2": "multi",
        "pca_mtgp": "multi", "pca_gp": "single", "peak_voxel": "single"}
PAIRS = [("gp", "mtgp"), ("gp", "mtgp_corr"), ("gp", "lmc"), ("ridge", "rrr"), ("lasso", "mtlasso"), ("pls1", "pls2")]


def make_model(name: str, seed: int = 0, **kw):
    if name == "gp":
        return SingleOutputGPs(seed=seed, **kw)
    if name == "mtgp":
        return MultiTaskGP(n_latent=1, seed=seed, **kw)
    if name == "mtgp_corr":
        return MultiTaskGP(n_latent=1, correlated_noise=True, seed=seed, **kw)
    if name == "lmc":
        return MultiTaskGP(n_latent=kw.pop("n_latent", 2), seed=seed, **kw)
    if name == "gp_augmented":
        return AugmentedSingle(lambda: SingleOutputGPs(seed=seed, **kw))
    if name == "ridge":
        return IndependentRidge()
    if name == "rrr":
        return ReducedRankRidge(seed=seed, **kw)
    if name == "lasso":
        return IndependentLasso()
    if name == "mtlasso":
        return MultiTaskLasso()
    if name == "pls1":
        return PLS1(seed=seed, **kw)
    if name == "pls2":
        return PLS2(seed=seed, **kw)
    if name in ("pca_mtgp", "pca_gp"):
        from .brain import PCAModel
        return PCAModel(base=name.split("_")[1], seed=seed, **kw)
    if name == "peak_voxel":
        from .brain import PeakVoxelModel
        return PeakVoxelModel(**kw)
    raise ValueError(f"unknown model '{name}'; choose from {', '.join(KIND)}")

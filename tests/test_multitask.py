"""Tests on synthetic data with known structure. Run: pytest tests/ (about a minute)."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from multitask.evaluation import cross_validate, make_folds
from multitask.gp import SingleOutputGPs
from multitask.linear import (IndependentLasso, MultiTaskLasso, PLS1, PLS2, ReducedRankRidge)
from multitask.models import KIND, PAIRS, make_model
from multitask.mtgp import MultiTaskGP, predict_trajectory
from multitask.selection import ard_rankings, forward_selection
from multitask.synthetic import lmc_sample, mask_missing, shared_factors, toy_1d

ROOT = Path(__file__).resolve().parents[1]
FAST = {"n_restarts": 1, "max_iter": 150}


def _mean_r2(res, name):
    return res.scores().query("model == @name").R2.mean()


# ------------------------------------------------------------------ the multi-task GP itself
@pytest.mark.parametrize("case", [(1, False), (2, False), (3, False), (1, True), (2, True)])
def test_mtgp_gradients_match_finite_differences(case):
    Q, corr = case
    rng = np.random.default_rng(Q)
    m = MultiTaskGP(n_latent=Q, correlated_noise=corr)
    rows = np.repeat(np.arange(6), 3)
    m.n_features_, m.n_tasks_, m.rank_ = 2, 3, 3
    m.Xo_, m.to_, m.yo_, m.ro_ = rng.normal(size=(18, 2)), np.tile(np.arange(3), 6), rng.normal(size=18), rows
    m.same_row_ = (rows[:, None] == rows[None, :]).astype(float)
    th = m._initial(rng, True)
    g = m._nlml(th)[1]
    h = 1e-5
    fd = np.array([(m._nlml(th + h * e)[0] - m._nlml(th - h * e)[0]) / (2 * h) for e in np.eye(len(th))])
    assert np.abs(g - fd).max() < 1e-5 * max(1.0, np.abs(fd).max())


def test_mtgp_recovers_task_correlation():
    X, Y, info = lmc_sample(n=120, correlation=0.9, n_outputs=3, seed=0)
    C = MultiTaskGP(**FAST).fit(X, Y).task_correlation()
    off = C[~np.eye(3, dtype=bool)]
    assert off.min() > 0.6


def test_mtgp_finds_no_relation_between_independent_outputs():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 2))
    Y = np.c_[np.sin(2 * X[:, 0]), np.cos(2 * X[:, 1])] + 0.2 * rng.normal(size=(100, 2))
    C = MultiTaskGP(**FAST).fit(X, Y).task_correlation()
    assert abs(C[0, 1]) < 0.5


def test_mtgp_handles_missing_outputs_and_uses_known_ones():
    X, Y, _ = lmc_sample(n=100, correlation=0.95, n_outputs=3, noise=0.1, seed=1)
    Ym = mask_missing(Y, 0.4, seed=0)
    m = MultiTaskGP(**FAST).fit(X[:80], Ym[:80])
    assert m.yo_.size == np.isfinite(Ym[:80]).sum()
    Yp = Y[80:].copy()
    Yp[:, 0] = np.nan
    err_given = np.mean((m.predict_given(X[80:], Yp)[:, 0] - Y[80:, 0]) ** 2)
    err_inputs = np.mean((m.predict(X[80:])[:, 0] - Y[80:, 0]) ** 2)
    assert err_given < err_inputs
    mean, sd = m.predict(X[80:], return_std=True)
    assert mean.shape == sd.shape == (20, 3) and (sd > 0).all()


def test_predict_given_leaves_the_fitted_model_unchanged():
    X, Y, _ = lmc_sample(n=60, seed=2)
    m = MultiTaskGP(**FAST).fit(X, Y)
    before = m.predict(X[:5])
    m.predict_given(X[:5], np.where(np.eye(5, 3, dtype=bool), Y[:5], np.nan))
    assert np.allclose(before, m.predict(X[:5]))


def test_correlated_noise_captures_a_patient_factor():
    """A patient-level factor not explained by the inputs: only the patient's other outputs tell
    us about it. The correlated-noise MTGP should use it; independent GPs cannot."""
    X, Y, _ = shared_factors(n=80, relatedness=0.8, patient_factor=1.0, seed=3)
    res = cross_validate(["gp", "mtgp_corr"], X, Y, k=4, n_jobs=2, mode="fill_in",
                         model_args={"gp": FAST, "mtgp_corr": FAST})
    assert _mean_r2(res, "mtgp_corr") > _mean_r2(res, "gp") + 0.2
    last = res.compare("gp", "mtgp_corr", n_boot=300).iloc[-1]
    assert last.verdict == "multi better"


def test_lmc_does_not_lose_much_on_unrelated_outputs():
    """With unrelated outputs a multi-output model has nothing to gain. The 2-latent LMC loses little;
    the 1-latent ICM (mtgp) can lose more, because every output must share one set of length
    scales (with this dataset: R2 0.53 vs 0.68 for independent GPs; see README)."""
    X, Y, _ = shared_factors(n=100, relatedness=0.0, seed=4)
    res = cross_validate(["gp", "lmc"], X, Y, k=4, model_args={"gp": FAST, "lmc": FAST})
    assert _mean_r2(res, "lmc") > _mean_r2(res, "gp") - 0.1


def test_trajectory_shape():
    X, Y = toy_1d(60, seed=0)
    m = MultiTaskGP(**FAST).fit(X, Y)
    mean, sd = predict_trajectory(m, X[0], np.linspace(0.1, 0.9, 7))
    assert mean.shape == sd.shape == (7, 3)


def test_single_output_gps_rank_relevant_features_first():
    X, Y, info = shared_factors(n=120, n_features=6, n_relevant=2, seed=5)
    rel = SingleOutputGPs(**FAST).fit(X, Y).relevance()
    assert rel.shape == (6, 4)
    top2 = np.argsort(-rel.mean(1))[:2]
    assert set(top2) == {0, 1}


# ------------------------------------------------------------------ linear pairs
def test_reduced_rank_ridge_finds_the_true_rank():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 8))
    B = rng.normal(size=(8, 1)) @ rng.normal(size=(1, 5))              # rank 1
    Y = X @ B + 0.3 * rng.normal(size=(200, 5))
    assert ReducedRankRidge().fit(X, Y).rank_ == 1


def test_multitask_lasso_shares_features_across_outputs():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(150, 20))
    Y = X[:, :3] @ rng.normal(size=(3, 4)) + 0.5 * rng.normal(size=(150, 4))
    S = MultiTaskLasso().fit(X, Y).selected()
    assert (S.all(1) | ~S.any(1)).all()                # a feature is used by all outputs or none
    assert S[:3].all()
    assert IndependentLasso().fit(X, Y).selected().shape == (20, 4)


def test_pls_choose_components_and_predict():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(100, 10))
    Y = X[:, :2] @ rng.normal(size=(2, 3)) + 0.3 * rng.normal(size=(100, 3))
    for m in (PLS1(), PLS2()):
        P = m.fit(X[:80], Y[:80]).predict(X[80:])
        assert P.shape == (20, 3)
        assert np.corrcoef(P[:, 0], Y[80:, 0])[0, 1] > 0.8


def test_joint_linear_models_drop_incomplete_rows_but_still_predict():
    X, Y, _ = shared_factors(n=80, seed=6)
    Ym = mask_missing(Y, 0.3, seed=1)
    for name in ("rrr", "mtlasso", "pls2", "ridge", "lasso", "pls1"):
        P = make_model(name).fit(X, Ym).predict(X)
        assert P.shape == Y.shape and np.isfinite(P).all()


@pytest.mark.parametrize("name", sorted(KIND))
def test_every_model_fits_and_predicts(name):
    X, Y, _ = shared_factors(n=50, n_features=5, seed=7)
    kw = FAST if name in ("gp", "mtgp", "mtgp_corr", "lmc", "gp_augmented", "pca_mtgp", "pca_gp") else {}
    m = make_model(name, **kw).fit(X, mask_missing(Y, 0.2, seed=2))
    assert m.predict(X[:4]).shape == (4, 4)
    Yp = Y[:4].copy()
    Yp[:, 0] = np.nan
    out = m.predict_given(X[:4], Yp, return_std=True)
    mean = out[0] if isinstance(out, tuple) else out
    assert mean.shape == (4, 4)


def test_pairs_are_single_then_multi():
    for s, m in PAIRS:
        assert KIND[s] == "single" and KIND[m] == "multi"


# ------------------------------------------------------------------ evaluation
def test_folds_partition_rows_and_keep_groups_together():
    folds = make_folds(30, k=5, repeats=2, seed=0)
    for r in (0, 1):
        test = np.concatenate([te for rr, _, te in folds if rr == r])
        assert sorted(test) == list(range(30))
    groups = np.repeat(np.arange(10), 3)
    for _, tr, te in make_folds(30, k=5, groups=groups):
        assert not set(groups[tr]) & set(groups[te])


def test_cross_validation_does_not_leak_test_outputs():
    """Changing the held-out rows' outputs must not change their predictions (from_inputs mode)."""
    X, Y, _ = shared_factors(n=40, n_features=4, seed=8)
    Y2 = Y.copy()
    folds = make_folds(40, k=4, seed=0)
    te = folds[0][2]
    Y2[te] += 100.0
    for name in ("ridge", "mtlasso"):
        a = cross_validate([name], X, Y, k=4).mean_prediction(name)[te]
        b = cross_validate([name], X, Y2, k=4).mean_prediction(name)[te]
        assert np.allclose(a, b)


def test_fill_in_mode_never_sees_the_output_being_predicted():
    """In fill-in mode, the output being predicted is hidden: changing it (for test rows only)
    must not change its prediction."""
    X, Y, _ = shared_factors(n=40, n_features=3, seed=9)
    te = make_folds(40, k=4, seed=0)[0][2]
    Y2 = Y.copy()
    Y2[te, 0] += 50.0
    a = cross_validate(["mtgp"], X, Y, k=4, mode="fill_in", model_args={"mtgp": FAST}).mean_prediction("mtgp")
    b = cross_validate(["mtgp"], X, Y2, k=4, mode="fill_in", model_args={"mtgp": FAST}).mean_prediction("mtgp")
    assert np.allclose(a[te, 0], b[te, 0])


def test_pca_is_fitted_inside_folds():
    from multitask.brain import PCAModel
    X, Y, _ = shared_factors(n=40, n_features=30, seed=10)
    m = PCAModel(base="gp", **FAST).fit(X[:30], Y[:30])
    assert np.allclose(m.mean_, X[:30].mean(0))


def test_compare_reports_positive_difference_when_multi_is_better():
    rng = np.random.default_rng(0)
    X, Y, _ = shared_factors(n=60, seed=11)
    res = cross_validate(["ridge", "rrr"], X, Y, k=3)
    res.predictions["rrr"] = (Y + 0.01 * rng.normal(size=Y.shape))[:, :, None]     # a near-perfect "multi" model
    c = res.compare("ridge", "rrr", n_boot=200)
    assert (c.difference > 0).all() and c.iloc[-1].verdict == "multi better"


# ------------------------------------------------------------------ feature selection
def test_forward_selection_finds_relevant_features_without_duplicates():
    rng = np.random.default_rng(12)
    X = rng.normal(size=(100, 8))
    Y = X[:, :2] @ np.array([[1.0, 0.8, 0.3], [0.4, -0.7, 1.0]]) + 0.5 * rng.normal(size=(100, 3))
    D = ard_rankings(X, Y)
    path = forward_selection("ridge", X, Y, D, start=[int(D[0, 0])], max_features=6, k=5)
    feats, _ = path.best()
    assert len(feats) == len(set(feats))
    assert {0, 1} <= set(feats)
    assert len(feats) <= 4


def test_ard_rankings_force_first():
    X, Y, _ = shared_factors(n=60, n_features=5, seed=13)
    D = ard_rankings(X, Y, first=4)
    assert (D[0] == 4).all() and all(len(set(D[:, j])) == 5 for j in range(D.shape[1]))


# ------------------------------------------------------------------ original toy problem and CLI
def test_toy_problem_matches_the_original_definition():
    X, Y = toy_1d(500, seed=0)
    assert X.shape == (500, 1) and Y.shape == (500, 3)
    assert np.corrcoef(Y.T)[0, 1] > 0.5                  # T2 = T1 + noise
    assert abs(np.corrcoef(Y.T)[0, 2]) < np.corrcoef(Y.T)[0, 1]


def test_cli_compare(tmp_path):
    X, Y, _ = shared_factors(n=40, n_features=3, n_outputs=2, seed=14)
    t = pd.DataFrame(np.c_[X, Y], columns=["a", "b", "c", "y1", "y2"])
    t.to_csv(tmp_path / "d.csv", index=False)
    r = subprocess.run([sys.executable, "-m", "multitask", "compare", str(tmp_path / "d.csv"), "--inputs", "a", "b", "c",
                        "--outputs", "y1", "y2", "--models", "ridge", "rrr", "--k", "4", "--out", str(tmp_path / "o")],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "o" / "scores.csv").exists() and (tmp_path / "o" / "comparisons.csv").exists()


def test_small_benchmark_runs():
    from multitask import benchmark
    scores, comps = benchmark.run(["toy_1d"], models=["ridge", "rrr"], datasets=1, k=3, verbose=False)
    assert set(scores.model) == {"ridge", "rrr"} and len(comps) == 1

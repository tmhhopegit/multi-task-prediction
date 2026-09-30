"""Synthetic multi-output regression problems with known structure.

    toy_1d            the original TestMultiTaskLearning.m problem: one input, three outputs
                      (two related, one not)
    shared_factors    outputs built from a few shared nonlinear functions of a few relevant inputs,
                      plus output-specific noise; `relatedness` (0-1) sets how much of each output
                      comes from the shared functions rather than its own private function
    lmc_sample        a draw from a multi-output GP prior with a chosen between-output correlation
    mask_missing      set a fraction of outputs to missing (NaN) at random, keeping at least one
                      observed output per row
"""
from __future__ import annotations

import numpy as np


def toy_1d(n: int = 100, seed: int = 0):
    """TestMultiTaskLearning.m: x ~ U(0,1); T1 = sin(x) + U(0,1); T2 = T1 + U(0,1); T3 = log(x) + U(0,1)."""
    rng = np.random.default_rng(seed)
    x = rng.random(n)
    t1 = np.sin(x) + rng.random(n)
    t2 = t1 + rng.random(n)
    t3 = np.log(x) + rng.random(n)
    return x[:, None], np.c_[t1, t2, t3]


def _smooth_function(rng, Xs):
    """A random smooth nonlinear function of the columns of Xs."""
    w = rng.normal(size=Xs.shape[1])
    a, b = rng.normal(size=2)
    z = Xs @ w / np.sqrt(Xs.shape[1])
    return np.sin(1.5 * z + a) + 0.5 * np.tanh(z * b) + 0.3 * z


def shared_factors(n: int = 200, n_features: int = 10, n_outputs: int = 4, n_relevant: int = 3,
                   n_factors: int = 2, relatedness: float = 0.8, noise: float = 0.5, patient_factor: float = 0.0,
                   seed: int = 0):
    """Returns X (n x features), Y (n x outputs), and a dict describing the truth.

    Each output = sqrt(relatedness) * (mix of the shared factor functions)
                + sqrt(1 - relatedness) * (its own private function) + noise.
    All factors and private functions depend only on the first n_relevant inputs.

    patient_factor: SD of a per-row latent (e.g. general severity) that is added to every output
    (with output-specific positive loadings) and is NOT a function of the inputs. Only a patient's
    other outputs carry information about it, so it is what makes "fill-in" prediction worthwhile."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_features))
    R = X[:, :n_relevant]
    factors = np.column_stack([_smooth_function(rng, R) for _ in range(n_factors)])
    factors = (factors - factors.mean(0)) / factors.std(0)
    mix = rng.normal(size=(n_factors, n_outputs))
    mix /= np.linalg.norm(mix, axis=0, keepdims=True)
    shared = factors @ mix
    private = np.column_stack([_smooth_function(rng, R) for _ in range(n_outputs)])
    private = (private - private.mean(0)) / private.std(0)
    signal = np.sqrt(relatedness) * shared + np.sqrt(1 - relatedness) * private
    Y = signal + noise * rng.normal(size=signal.shape)
    if patient_factor:
        u = rng.normal(size=n)
        load = rng.uniform(0.7, 1.3, size=n_outputs)
        Y = Y + patient_factor * u[:, None] * load[None, :]
    return X, Y, {"signal": signal, "relevant": list(range(n_relevant)), "signal_correlation": np.corrcoef(signal.T)}


def lmc_sample(n: int = 150, n_features: int = 2, correlation: float = 0.9, n_outputs: int = 3,
               lengthscale: float = 1.0, noise: float = 0.3, seed: int = 0):
    """One draw from a multi-output GP with between-output correlation `correlation` (same for all
    pairs) and a squared-exponential kernel over the inputs."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_features))
    d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    Kx = np.exp(-d2 / (2 * lengthscale ** 2))
    B = np.full((n_outputs, n_outputs), correlation) + (1 - correlation) * np.eye(n_outputs)
    K = np.kron(B, Kx) + 1e-8 * np.eye(n * n_outputs)
    f = np.linalg.cholesky(K) @ rng.normal(size=n * n_outputs)
    F = f.reshape(n_outputs, n).T
    return X, F + noise * rng.normal(size=F.shape), {"signal": F, "B": B}


def mask_missing(Y, fraction: float = 0.3, seed: int = 0, keep_one: bool = True):
    """Copy of Y with `fraction` of entries set to NaN at random (at least one observed per row)."""
    rng = np.random.default_rng(seed)
    Y = np.array(Y, dtype=np.float64)
    miss = rng.random(Y.shape) < fraction
    if keep_one:
        for i in np.flatnonzero(miss.all(1)):
            miss[i, rng.integers(Y.shape[1])] = False
    Y[miss] = np.nan
    return Y
